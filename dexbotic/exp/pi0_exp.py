import argparse
import atexit
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from typing import Optional

import megfile
import torch
import numpy as np
from easydict import EasyDict
from torch.utils.data import DataLoader
from tqdm import tqdm
from flask import Flask, jsonify, request
from loguru import logger
from PIL import Image
from transformers import AutoTokenizer, AutoImageProcessor, BaseImageProcessor
import transformers

from dexbotic.exp.base_exp import (
    ActionConfig,
    BaseExp,
    TokenizerConfig,
    ComputeNormActionConfig,
    Config,
    DataConfig,
    ModelConfig,
    TrainerConfig,
)
from dexbotic.exp.pi0_optimizer import Pi0OptimizerConfig
from dexbotic.exp.pi0_interpretability import Pi0OnlineRouteCollector
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM, Pi0Model
from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddAction,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import (
    Pipeline,
    ToDict,
    ToList,
    ToNumpy,
    ToTensor,
)
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.data.dataset.transform.output import AbsoluteAction, ActionDenorm
from dexbotic.data.dataset.dex_dataset import DexDataset
from .base_exp import OPENAI_CLIP_PATH
from dexbotic.data.dataset.rgb_preprocess import DummyRGBProcessor
from dexbotic.data.dataset.tokenization import DummyTokenization
from dexbotic.tokenization.process import Pi0Tokenization
import dexbotic.data.utils.normalize as normalize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, unknown = parser.parse_known_args()
    return args


def _env_truthy(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _format_module_details(module: torch.nn.Module) -> str:
    details = []
    if hasattr(module, "use_moe"):
        details.append(f"use_moe={bool(getattr(module, 'use_moe', False))}")
        if hasattr(module, "moe_type"):
            details.append(f"moe_type={getattr(module, 'moe_type')}")
        if hasattr(module, "num_experts"):
            details.append(f"num_experts={getattr(module, 'num_experts')}")
        if hasattr(module, "moe_top_k"):
            details.append(f"moe_top_k={getattr(module, 'moe_top_k')}")
        elif hasattr(module, "top_k"):
            details.append(f"top_k={getattr(module, 'top_k')}")
        if hasattr(module, "use_general_expert"):
            details.append(
                f"use_general_expert={bool(getattr(module, 'use_general_expert'))}"
            )
        if hasattr(module, "general_expert_weight"):
            details.append(
                f"general_expert_weight={getattr(module, 'general_expert_weight')}"
            )
    elif hasattr(module, "num_experts") and hasattr(module, "top_k"):
        details.append(f"num_experts={getattr(module, 'num_experts')}")
        if hasattr(module, "moe_top_k"):
            details.append(f"moe_top_k={getattr(module, 'moe_top_k')}")
        else:
            details.append(f"top_k={getattr(module, 'top_k')}")

    if details:
        return " [" + ", ".join(details) + "]"
    return ""


def format_model_tree(module: torch.nn.Module, max_depth: int = 4) -> str:
    lines = [module.__class__.__name__]

    def _walk(mod: torch.nn.Module, prefix: str, depth: int) -> None:
        if depth >= max_depth:
            return
        children = list(mod.named_children())
        for idx, (name, child) in enumerate(children):
            is_last = idx == len(children) - 1
            connector = "\\--" if is_last else "|--"
            details = _format_module_details(child)
            lines.append(
                f"{prefix}{connector} {name}: {child.__class__.__name__}{details}"
            )
            next_prefix = prefix + ("   " if is_last else "|  ")
            _walk(child, next_prefix, depth + 1)

    _walk(module, "", 0)
    return "\n".join(lines)


@dataclass
class Pi0TrainerConfig(TrainerConfig):
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=10000)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    gradient_checkpointing: bool = field(default=True)
    model_max_length: int = field(default=48)
    dataloader_num_workers: int = field(default=16)
    logging_steps: int = field(default=1)
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    task_aware_grad_accum: bool = field(default=True)
    pcgrad_grouping: str = field(default="prompt_hash")
    pcgrad_on_shared: bool = field(default=True)
    shared_param_group_names: list[str] = field(
        default_factory=lambda: [
            "base",
            "moe",
            "router",
            "moe_base",
            "mm_projector",
            "mm_vision",
            "action_head",
        ]
    )
    shared_expert_group_names: list[str] = field(
        default_factory=lambda: ["moe_base"]
    )
    shared_expert_freeze_steps: int = field(default=0)
    gradient_conflict_stats_enable: bool = field(default=True)
    gradient_conflict_stats_steps: int = field(default=1)
    gradient_conflict_stats_output_dir: str = field(default="")
    feature_vis_enable: bool = field(default=False)
    feature_vis_steps: int = field(default=2000)
    feature_vis_num_samples: int = field(default=320)
    feature_vis_max_scan: int = field(default=4000)
    feature_vis_output_dir: str = field(default="")


class Pi0ComputeNormActionConfig(ComputeNormActionConfig):
    def compute_norm_stats(self, dataset_name: str) -> None:
        self.norm_save_path = os.path.join(
            os.path.dirname(self.norm_save_path),
            hashlib.md5(dataset_name.encode()).hexdigest()[:8],
        )
        dataset_name_list = dataset_name.split("+")
        action_process_func = self.build_action_process_func()
        dataset_list = self._get_dataset(action_process_func, dataset_name_list)
        norm_files = {}

        for dataset_name, dataset in dataset_list:
            norm_file = self._process_one_dataset(dataset_name, dataset)
            norm_files[dataset_name] = (norm_file, dataset.dataset_map[0])

        self._merge_norm_stats(norm_files)

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(),
            ]
        )

        return action_config

    def _get_dataset(self, action_process_func, dataset_name_list):
        robot_dataset_list = []
        for dataset_name in dataset_name_list:
            robot_dataset = DexDataset(
                data_args=EasyDict(
                    dataset_name=dataset_name,
                    num_images=1,
                    data_keys=["action", "state"],
                    image_processor=AutoImageProcessor.from_pretrained(
                        OPENAI_CLIP_PATH
                    ),
                    image_aspect_ratio=None,
                    aug_policy=None,
                ),
                tokenization_func=DummyTokenization(),
                action_process_func=action_process_func,
                image_process_func=DummyRGBProcessor(),
            )
            robot_dataset_list.append((dataset_name, robot_dataset))
        return robot_dataset_list

    def _process_one_dataset(self, dataset_name, dataset):
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=64)

        norm_keys = ["state", "action"]
        stats = {key: normalize.RunningStats() for key in norm_keys}
        for batch_idx, batch in tqdm(
            enumerate(dataloader), desc="Computing norm stats"
        ):
            if batch_idx > 1000:
                break
            for key in norm_keys:
                values = batch[key].numpy()
                stats[key].update(values.reshape(-1, values.shape[-1]))
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

        save_path = os.path.join(self.norm_save_path, dataset_name)
        logger.info(f"Saving norm stats to {save_path}")
        normalize.save(save_path, norm_stats)

        return os.path.join(save_path, "norm_stats.json")

    def _merge_norm_stats(
        self, norm_files, per_task_norm=False, norm_keys=["action", "state"]
    ):
        norm_stats = {
            "default": {"min": -1, "max": 1},
        }
        for norm_key in norm_keys:
            min_list = []
            max_list = []
            mean_list = []
            std_list = []
            for dataset_name, (norm_file, dataset_path) in norm_files.items():
                with open(norm_file, "r") as f:
                    stats = json.load(f)["norm_stats"][norm_key]
                if per_task_norm:
                    norm_stats[dataset_path] = {
                        "default": {
                            "min": stats["q01"],
                            "max": stats["q99"],
                            "mean": stats["mean"],
                            "std": stats["std"],
                        }
                    }
                min_list.append(stats["q01"])
                max_list.append(stats["q99"])
                mean_list.append(stats["mean"])
                std_list.append(stats["std"])
            min_list = np.array(min_list).min(axis=0).tolist()
            max_list = np.array(max_list).max(axis=0).tolist()
            mean_list = np.array(mean_list).mean(axis=0).tolist()
            std_list = np.array(std_list).mean(axis=0).tolist()
            norm_stats[norm_key] = {
                "min": min_list,
                "max": max_list,
                "mean": mean_list,
                "std": std_list,
            }

        with open(os.path.join(self.norm_save_path, "norm_stats.json"), "w") as f:
            json.dump({"norm_stats": norm_stats}, f, indent=2)


@dataclass
class Pi0ActionConfig(ActionConfig):
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                AddAction(predict_length=1),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ActionNorm(statistic_mapping=statistic_mapping),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )

        return action_config


@dataclass
class Pi0DataConfig(DataConfig):
    num_images: int = field(default=3)
    data_keys: list[str] = field(
        default_factory=lambda: [
            "input_ids",
            "labels",
            "action",
            "image",
            "state",
            "image_masks",
        ]
    )
    aug_policy: str | list[str] = field(
        default_factory=lambda: ["pi0", "color", "color"]
    )
    action_config: Pi0ActionConfig = field(default_factory=Pi0ActionConfig)
    image_pad_mode: str = field(default="zero")
    report_task_prompts: bool = field(default=False)
    task_id_from_prompt: bool = field(default=True)
    task_prompt_max_items: int = field(default=50)

    def _build_dataset(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        chat_template: str,
        image_processor: BaseImageProcessor,
    ) -> DexDataset:
        # FIXME: DO NOT USE EASYDICT IN NEXT VERSION
        data_args = EasyDict(
            {
                "dataset_name": self.dataset_name,
                "num_images": self.num_images,
                "data_keys": self.data_keys,
                "images_keys": self.images_keys,
                "aug_policy": self.aug_policy,
                "image_aspect_ratio": self.image_aspect_ratio,
                "image_processor": image_processor,
                "chat_template": chat_template,
                "image_pad_mode": self.image_pad_mode,
                "report_task_prompts": self.report_task_prompts,
                "task_id_from_prompt": self.task_id_from_prompt,
                "task_prompt_max_items": self.task_prompt_max_items,
            }
        )
        action_process_func = self.action_config.build_action_process_func()
        tokenization_func = Pi0Tokenization(tokenizer, data_args)
        dataset = DexDataset(
            data_args=data_args,
            tokenization_func=tokenization_func,
            action_process_func=action_process_func,
        )
        return dataset


@dataclass
class Pi0ModelConfig(ModelConfig):
    """
    Pi0 模型配置类 - 继承自基础模型配置
    """

    model_name_or_path: str = field(default="./checkpoints/Dexbotic-PI0")
    moe_weight_path: Optional[str] = field(default=None)
    moe_noise_std: float = field(default=0.0)
    moe_gating_init_std: float = field(default=0.006)
    use_moe: Optional[bool] = field(default=None)
    moe_type: Optional[str] = field(default=None)
    num_experts: Optional[int] = field(default=None)
    moe_top_k: Optional[int] = field(default=None)
    use_general_expert: Optional[bool] = field(default=None)
    general_expert_weight: Optional[float] = field(default=None)
    moe_lora_rank: Optional[int] = field(default=None)
    moe_lora_alpha: Optional[float] = field(default=None)
    moe_load_balancing_weight: Optional[float] = field(default=None)
    moe_stable: Optional[bool] = field(default=None)
    task_token_enable: Optional[bool] = field(default=True)
    task_token_use_task_id: Optional[bool] = field(default=True)
    task_token_from_text: Optional[bool] = field(default=True)
    task_token_pooling: Optional[str] = field(default="mean")
    task_token_num_tasks: Optional[int] = field(default=None)
    print_model_structure: bool = field(default=False)
    model_structure_depth: int = field(default=6)

    def build_model(self) -> Pi0ForCausalLM:
        moe_overrides = {
            "use_moe": self.use_moe,
            "moe_type": self.moe_type,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
            "use_general_expert": self.use_general_expert,
            "general_expert_weight": self.general_expert_weight,
            "moe_load_balancing_weight": self.moe_load_balancing_weight,
        }
        moe_type_norm = str(self.moe_type).lower().replace("_", "-")
        if moe_type_norm == "moe-lora":
            moe_overrides["moe_lora_rank"] = self.moe_lora_rank
            moe_overrides["moe_lora_alpha"] = self.moe_lora_alpha
        if self.moe_stable is not None:
            os.environ["DEXBOTIC_MOE_STABLE"] = "1" if self.moe_stable else "0"
        elif moe_type_norm == "moe-lora":
            os.environ.setdefault("DEXBOTIC_MOE_STABLE", "1")
        task_token_overrides = {
            "task_token_enable": self.task_token_enable,
            "task_token_use_task_id": self.task_token_use_task_id,
            "task_token_from_text": self.task_token_from_text,
            "task_token_pooling": self.task_token_pooling,
            "task_token_num_tasks": self.task_token_num_tasks,
        }
        moe_overrides.update(
            {k: v for k, v in task_token_overrides.items() if v is not None}
        )
        moe_overrides = {k: v for k, v in moe_overrides.items() if v is not None}
        model = Pi0ForCausalLM.from_pretrained(
            self.model_name_or_path, **moe_overrides
        )
        if self.moe_weight_path is not None:
            from dexbotic.model.pi0.moe_weight_loader import MoEWeightLoader

            loader = MoEWeightLoader(
                params_path=self.moe_weight_path,
                noise_std=self.moe_noise_std,
                gating_init_std=self.moe_gating_init_std,
            )
            loader.load(model, strict=False)
        moe_type_norm = str(getattr(model.config, "moe_type", "")).lower().replace("_", "-")
        if moe_type_norm == "moe-lora":
            reset_count = model.reset_moe_lora_params()
            if reset_count:
                print(f"[pi0] Reset MoE-LoRA params for {reset_count} layers")
        if self.print_model_structure or _env_truthy("DEXBOTIC_PRINT_MODEL"):
            depth = int(os.getenv("DEXBOTIC_MODEL_DEPTH", self.model_structure_depth))
            logger.info(
                "[pi0] Model structure (depth=%s):\n%s",
                depth,
                format_model_tree(model, max_depth=depth),
            )
        return model


@dataclass
class Pi0TokenizerConfig(TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi0InferenceConfig(Config):
    model_name_or_path: Optional[str] = field(default=None)
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [12, 13])
    action_dim: int = field(default=14)
    interpretability_mode: str = field(
        default_factory=lambda: os.getenv("DEXBOTIC_INTERPRETABILITY_MODE", "none")
    )
    interpretability_output_dir: str = field(
        default_factory=lambda: os.getenv(
            "DEXBOTIC_INTERPRETABILITY_OUTPUT_DIR",
            "./interpretability_outputs/online",
        )
    )
    interpretability_model_label: str = field(
        default_factory=lambda: os.getenv("DEXBOTIC_INTERPRETABILITY_MODEL_LABEL", "online")
    )
    interpretability_flush_steps: int = field(
        default_factory=lambda: _env_int("DEXBOTIC_INTERPRETABILITY_FLUSH_STEPS", 10)
    )
    interpretability_save_token_routes_limit_batches: int = field(
        default_factory=lambda: _env_int(
            "DEXBOTIC_INTERPRETABILITY_SAVE_TOKEN_ROUTES_LIMIT_BATCHES", 32
        )
    )
    interpretability_save_moe_raw_limit_batches: int = field(
        default_factory=lambda: _env_int(
            "DEXBOTIC_INTERPRETABILITY_SAVE_MOE_RAW_LIMIT_BATCHES", 0
        )
    )

    def run(self) -> None:
        self._initialize_inference()
        self.app = Flask(__name__)
        self.app.add_url_rule(
            "/process_frame", "process_frame", self.process_frame, methods=["POST"]
        )
        self.app.add_url_rule(
            "/interpretability/save", "interpretability_save", self.process_interpretability_save,
            methods=["GET", "POST"],
        )
        self.app.add_url_rule(
            "/interpretability/reset", "interpretability_reset", self.process_interpretability_reset,
            methods=["POST"],
        )
        self.app.run(host="0.0.0.0", port=self.port, debug=False, threaded=False)

    def _initialize_inference(self) -> None:
        if self.norm_stats is None:
            norm_stats_file = os.path.join(self.model_name_or_path, "norm_stats.json")
            self.norm_stats = self.read_normalization_stats(norm_stats_file)
        logger.info(f"Normalization stats: {self.norm_stats}")

        self._load_model()
        self.prev_text = None
        self.timestep = 0
        self.episode = 0
        self._initialize_online_interpretability()

    def _online_interpretability_enabled(self) -> bool:
        mode = str(self.interpretability_mode).strip().lower()
        return mode == "online" or _env_truthy("DEXBOTIC_ONLINE_INTERPRETABILITY")

    def _initialize_online_interpretability(self) -> None:
        self.online_interpretability_collector = None
        if not self._online_interpretability_enabled():
            logger.info("Online interpretability disabled")
            return
        self.online_interpretability_collector = Pi0OnlineRouteCollector(
            self.model,
            output_dir=self.interpretability_output_dir,
            model_label=self.interpretability_model_label,
            flush_steps=self.interpretability_flush_steps,
            save_token_routes_limit_batches=(
                self.interpretability_save_token_routes_limit_batches
            ),
            save_moe_raw_limit_batches=self.interpretability_save_moe_raw_limit_batches,
        )
        atexit.register(self._save_online_interpretability_at_exit)
        logger.info(
            "Online interpretability enabled: output_dir={}, flush_steps={}",
            self.interpretability_output_dir,
            self.interpretability_flush_steps,
        )

    def _save_online_interpretability_at_exit(self) -> None:
        collector = getattr(self, "online_interpretability_collector", None)
        if collector is None:
            return
        try:
            collector.save()
        except Exception as exc:
            logger.warning("Failed to save online interpretability at exit: {}", exc)

    def process_interpretability_save(self):
        collector = getattr(self, "online_interpretability_collector", None)
        if collector is None:
            return jsonify({"enabled": False, "status": "disabled"})
        try:
            summary = collector.save()
            summary["enabled"] = True
            return jsonify(summary)
        except Exception as exc:
            return jsonify({"enabled": True, "status": "error", "error": str(exc)}), 500

    def process_interpretability_reset(self):
        collector = getattr(self, "online_interpretability_collector", None)
        if collector is None:
            return jsonify({"enabled": False, "status": "disabled"})
        collector.reset()
        return jsonify({"enabled": True, "status": "reset"})


    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = Pi0ForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
        ).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name_or_path, use_fast=False
        )
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model.config
        self.tokenization_func = Pi0Tokenization(self.tokenizer)
        logger.info("Model loaded successfully")

        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                ActionNorm(statistic_mapping=self.norm_stats, strict=False),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False),
                AbsoluteAction(),
            ]
        )

    def read_normalization_stats(self, action_norm_file):
        logger.info(f"Reading normalization stats from {action_norm_file}")
        if action_norm_file is None or not megfile.smart_exists(action_norm_file):
            return {"min": -1, "max": 1}
        with megfile.smart_open(action_norm_file, "r") as f:
            norm_stats = json.load(f)
            if "norm_stats" in norm_stats:
                norm_stats = norm_stats["norm_stats"]
        return ToNumpy()(norm_stats)

    def process_frame(self) -> None:
        results = self._get_response(
            text=request.form.get("text", ""),
            images=request.files.getlist("image", None),
            states=request.form.get("states", None),
            batch_size=request.form.get("batch_size", 1),
        )
        return jsonify({"response": results})

    def _get_response(
        self,
        text: str | list[str],
        images: list[str],
        states: Optional[str | list[str]] = None,
        batch_size: int = 1,
    ) -> str:
        t0 = time.monotonic()
        batch_size = int(batch_size)
        assert len(images) % batch_size == 0, (
            f"Number of images {len(images)} is not divisible by batch size {batch_size}"
        )
        num_images = len(images) // batch_size
        images = [
            images[i * num_images : (i + 1) * num_images] for i in range(batch_size)
        ]
        if isinstance(text, str):
            text = [text] * batch_size

        batch_images = [
            [Image.open(i).convert("RGB") for i in image_items]
            for image_items in images
        ]
        batch_images_tensor = [
            self.model.process_images(image_items).to(dtype=self.model.dtype)
            for image_items in batch_images
        ]

        if num_images != self.num_images:
            batch_images_tensor = [
                torch.cat(
                    [
                        image_tensor,
                        torch.zeros_like(image_tensor[0:1]).repeat(
                            self.num_images - num_images, 1, 1, 1
                        ),
                    ],
                    dim=0,
                )
                if len(image_tensor) < self.num_images
                else image_tensor[: self.num_images]
                for image_tensor in batch_images_tensor
            ]

        batch_image_masks = [
            torch.tensor(
                [True for _ in range(num_images)]
                + [False for _ in range(self.num_images - num_images)],
                device=image_tensor.device,
            )
            for image_tensor in batch_images_tensor
        ]
        batch_images_tensor = torch.stack(batch_images_tensor, dim=0)
        batch_image_masks = torch.stack(batch_image_masks, dim=0)

        self._save_image(batch_images[0], text[0])

        prompt = text
        batch_input_ids = np.array(
            [self.tokenization_func([{"value": p}])["input_ids"] for p in prompt]
        )
        batch_attention_mask = np.array(
            [np.array(ids != self.tokenizer.pad_token_id) for ids in batch_input_ids]
        )

        if states is not None:
            if isinstance(states, str):
                batch_states = np.array(json.loads(states))
                if batch_states.ndim == 1:
                    batch_states = batch_states[None]
                assert batch_states.shape[0] == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got length {len(batch_states)}."
                )
            elif isinstance(states, (list, tuple)) and all(
                isinstance(s, str) for s in states
            ):
                assert len(states) == batch_size, (
                    f"Batch inference requires states to be a list with length {batch_size}, "
                    f"but got {type(states)} with length {len(states)}."
                )
                batch_states = [json.loads(s) for s in states]
                batch_states = np.array(batch_states)
        else:
            batch_states = np.zeros(
                (
                    batch_size,
                    self.model.model.config.action_dim,
                ),
                dtype=np.float32,
            )

        inference_args = {
            "input_ids": batch_input_ids,
            "attention_mask": batch_attention_mask,
            "images": batch_images_tensor,
            "image_masks": batch_image_masks,
            "state": batch_states,
            "meta_data": {
                "non_delta_mask": np.array(self.non_delta_mask),
            },
        }

        inputs = self.input_transform(inference_args)
        inputs["states"] = inputs["state"]
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        collector = getattr(self, "online_interpretability_collector", None)
        if collector is not None:
            collector.start_batch(text)
        try:
            actions = self.model.inference_action(**inputs)
        finally:
            if collector is not None:
                collector.end_batch()
                collector.maybe_flush()
                
        outputs = {
            k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }
        outputs["action"] = actions.detach().cpu().numpy()
        outputs = self.output_transform(outputs)
        logger.info(f"Processing time: {time.monotonic() - t0}")
        return outputs["action"][0, ..., : self.action_dim].tolist()

    def _save_image(self, images: list[Image.Image], text: str) -> None:
        if not self.save_image:
            return
        if text == self.prev_text:
            self.timestep += 1
        else:
            self.timestep = 0
            self.prev_text = text
            self.episode += 1
        save_image_dir_episode = os.path.join(self.save_image_dir, str(self.episode))
        os.makedirs(save_image_dir_episode, exist_ok=True)
        for idx, image in enumerate(images):
            image.save(
                os.path.join(save_image_dir_episode, f"{self.timestep}_{idx}.png")
            )
        if self.timestep == 0:
            with open(os.path.join(save_image_dir_episode, "text.txt"), "w") as f:
                f.write(text)


@dataclass
class Pi0Exp(BaseExp):
    model_config: Pi0ModelConfig = field(default_factory=Pi0ModelConfig)
    optimizer_config: Pi0OptimizerConfig = field(default_factory=Pi0OptimizerConfig)
    trainer_config: Pi0TrainerConfig = field(default_factory=Pi0TrainerConfig)
    data_config: Pi0DataConfig = field(default_factory=Pi0DataConfig)
    tokenizer_config: Pi0TokenizerConfig = field(default_factory=Pi0TokenizerConfig)
    inference_config: Pi0InferenceConfig = field(default_factory=Pi0InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi0ComputeNormActionConfig()
        self.data_config.action_config.compute_norm_stats(self.data_config.dataset_name)

    def _auto_compute_norm_stats(self) -> None:
        if (
            not self.data_config.auto_norm
            or self.data_config.action_config.statistic_mapping is not None
        ):
            return
        if self.local_rank == 0:
            print(
                f"Action config before auto compute norm: {self.data_config.action_config}"
            )
        _action_config = self.data_config.action_config
        norm_config = Pi0ComputeNormActionConfig()
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing norm stats on rank0")
            self.compute_norm_stats()
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        _action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = _action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = Pi0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
