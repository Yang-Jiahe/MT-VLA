import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import megfile
import torch
from loguru import logger
from transformers import AutoTokenizer

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
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
from dexbotic.data.dataset.transform.output import ActionDenorm
from dexbotic.exp.pi0_exp import Pi0ActionConfig as _Pi0ActionConfig
from dexbotic.exp.pi05_exp import (
    Pi0ComputeNormActionConfig as _Pi0ComputeNormActionConfig,
)
from dexbotic.exp.pi05_exp import Pi0DataConfig as _Pi0DataConfig
from dexbotic.exp.pi05_exp import Pi0InferenceConfig as _Pi0InferenceConfig
from dexbotic.exp.pi05_exp import Pi0OptimizerConfig as _Pi0OptimizerConfig
from dexbotic.exp.pi05_exp import Pi0TokenizerConfig as _Pi0TokenizerConfig
from dexbotic.exp.pi05_exp import Pi0TrainerConfig as _Pi0TrainerConfig
from dexbotic.exp.pi05_exp import Pi05Exp as _Pi05Exp
from dexbotic.exp.pi05_exp import Pi05ModelConfig as _Pi05ModelConfig
from dexbotic.exp.pi0_exp import _env_truthy, format_model_tree
from dexbotic.model.pi05.pi05_arch import Pi05ForCausalLM
from dexbotic.model.pi0.moe_weight_loader import MoEWeightLoader
from dexbotic.tokenization.process import Pi0Tokenization


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


@dataclass
class Pi05OptimizerConfig(_Pi0OptimizerConfig):
    base_lr: float = field(default=5e-5)
    moe_lr: float = field(default=2.5e-5)
    router_lr: float = field(default=5e-5)
    moe_lora_lr: float = field(default=1e-4)
    moe_base_lr: float = field(default=2.5e-5)

    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=10_000)
    moe_warmup_steps: int = field(default=1000)
    router_warmup_steps: int = field(default=100)
    moe_lora_warmup_steps: int = field(default=1000)
    moe_base_warmup_steps: int = field(default=1000)

    weight_decay: float = field(default=1e-6)
    moe_weight_decay: float = field(default=1e-6)
    router_weight_decay: float = field(default=1e-6)
    moe_lora_weight_decay: float = field(default=1e-6)
    moe_base_weight_decay: float = field(default=1e-6)

    decay_steps: int = field(default=30000)
    moe_decay_steps: int = field(default=30000)
    router_decay_steps: int = field(default=30000)
    moe_lora_decay_steps: int = field(default=30000)
    moe_base_decay_steps: int = field(default=30_000)

    decay_lr: float = field(default=1e-6)
    moe_decay_lr: float = field(default=1e-6)
    router_decay_lr: float = field(default=1e-6)
    moe_lora_decay_lr: float = field(default=1e-6)
    moe_base_decay_lr: float = field(default=1e-6)


@dataclass
class Pi05TrainerConfig(_Pi0TrainerConfig):
    moe_type = os.getenv("MOE_TYPE", "")
    wandb_project: str = field(default="dexbotic-pi05-libero-all")
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=120000)
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=8)
    per_device_train_batch_size: int = field(default=8)#8
    gradient_accumulation_steps: int = field(default=4)#1
    model_max_length: int = field(default=200)
    output_dir: str = field(
        default=f"/mlp_vepfs/share/yjh/checkpoints/mtvla7/libero_pi05/{moe_type}/84_task_{datetime.now().strftime('%m%d')}"
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(
        default_factory=lambda: {"min_lr": 5e-5}
    )  # 5e-5 -> 5e-5
    task_aware_grad_accum: bool = field(default=True)
    pcgrad_on_shared: bool = field(default=True)
    pcgrad_grouping: str = field(default="prompt_hash")
    shared_expert_freeze_steps: int = field(default=20000)


class Pi05ComputeNormActionConfig(_Pi0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=10, flatten=False, padding_mode="last"),
                ToList(),
            ]
        )

        return action_config


@dataclass
class Pi05ActionConfig(_Pi0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=10)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=10, flatten=False, padding_mode="last"),
                ActionNorm(statistic_mapping=statistic_mapping),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class Pi05DataConfig(_Pi0DataConfig):
    dataset_name: str = field(default="libero_pi0_all")
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
        default_factory=lambda: ["pi0", "color", "identity"]
    )
    action_config: Pi05ActionConfig = field(default_factory=Pi05ActionConfig)
    report_task_prompts: bool = field(default=True)
    task_id_from_prompt: bool = field(default=False)
    task_prompt_max_items: int = field(default=50)


@dataclass
class Pi05ModelConfig(_Pi05ModelConfig):
    model_name_or_path: str = field(default="/dexmal-fa-yjh-data/checkpoints/Dexbotic-PI05")
    use_moe: bool = field(default=True)
    moe_type: str = field(default=os.getenv("MOE_TYPE", ""))
    num_experts: int = field(default=4)
    moe_top_k: int = field(default=1)
    use_general_expert: bool = field(default=True)
    general_expert_weight: float = field(default=0.5)
    moe_lora_rank: int = field(default=4)
    moe_lora_alpha: Optional[float] = field(default=4)
    moe_load_balancing_weight: float = field(default=0.005)
    moe_stable: bool = field(default=True)
    moe_weight_path: Optional[str] = field(default="/mlp_vepfs/share/yjh/checkpoints/dexbotic/libero_all_pi05/open16-0320/model-00002-of-00002.safetensors")
    moe_noise_std: float = field(default=0.0)
    moe_gating_init_std: float = field(default=0.006)
    print_model_structure: bool = field(default=True)
    model_structure_depth: int = field(default=6)

    task_token_enable: bool = field(default=True)
    task_token_use_task_id: bool = field(default=False)
    task_token_from_text: bool = field(default=True)
    task_token_pooling: str = field(default="mean")
    task_token_num_tasks: Optional[int] = field(default=40)

    def build_model(self) -> Pi05ForCausalLM:
        moe_overrides = {
            "use_moe": self.use_moe,
            "moe_type": self.moe_type,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
            "use_general_expert": self.use_general_expert,
            "general_expert_weight": self.general_expert_weight,
            "moe_load_balancing_weight": self.moe_load_balancing_weight,
            "task_token_enable": self.task_token_enable,
            "task_token_use_task_id": self.task_token_use_task_id,
            "task_token_from_text": self.task_token_from_text,
            "task_token_pooling": self.task_token_pooling,
            "task_token_num_tasks": self.task_token_num_tasks,
        }
        moe_type_norm = str(self.moe_type).lower().replace("_", "-")
        if moe_type_norm == "moe-lora":
            moe_overrides["moe_lora_rank"] = self.moe_lora_rank
            moe_overrides["moe_lora_alpha"] = self.moe_lora_alpha
        moe_overrides = {k: v for k, v in moe_overrides.items() if v is not None}
        if self.moe_stable:
            os.environ["DEXBOTIC_MOE_STABLE"] = "1"
        print(
            "[pi05] MoE config:",
            {
                "use_moe": self.use_moe,
                "moe_type": self.moe_type,
                "num_experts": self.num_experts,
                "moe_top_k": self.moe_top_k,
                "use_general_expert": self.use_general_expert,
                "general_expert_weight": self.general_expert_weight,
                "moe_load_balancing_weight": self.moe_load_balancing_weight,
                "moe_weight_path": self.moe_weight_path,
                "moe_noise_std": self.moe_noise_std,
                "moe_gating_init_std": self.moe_gating_init_std,
                "moe_stable": self.moe_stable,
            },
        )
        if moe_type_norm == "moe-lora":
            print(
                "[pi05] MoE-Lora config:",
                {
                    "moe_lora_rank": self.moe_lora_rank,
                    "moe_lora_alpha": self.moe_lora_alpha,
                },
            )
        config = Pi05ForCausalLM.config_class.from_pretrained(
            self.model_name_or_path, **moe_overrides
        )
        action_cfg = getattr(config, "action_config", None)
        if action_cfg is not None:
            action_cfg.use_moe = self.use_moe
            action_cfg.moe_type = self.moe_type
            action_cfg.num_experts = self.num_experts
            action_cfg.moe_top_k = self.moe_top_k
            action_cfg.use_general_expert = self.use_general_expert
            action_cfg.general_expert_weight = self.general_expert_weight
            if moe_type_norm == "moe-lora":
                action_cfg.moe_lora_rank = self.moe_lora_rank
                action_cfg.moe_lora_alpha = self.moe_lora_alpha

        model = Pi05ForCausalLM.from_pretrained(self.model_name_or_path, config=config)
        model.model.config.chunk_size = 10
        if self.moe_weight_path is not None:
            print("[pi05] Loading MoE weights from", self.moe_weight_path)
            loader = MoEWeightLoader(
                params_path=self.moe_weight_path,
                noise_std=self.moe_noise_std,
                gating_init_std=self.moe_gating_init_std,
            )
            loader.load(model, strict=False)
        if moe_type_norm == "moe-lora":
            reset_count = model.reset_moe_lora_params()
            if reset_count:
                print(f"[pi05] Reset MoE-LoRA params for {reset_count} layers")
        if self.print_model_structure or _env_truthy("DEXBOTIC_PRINT_MODEL"):
            depth = int(os.getenv("DEXBOTIC_MODEL_DEPTH", self.model_structure_depth))
            print(f"[pi05] Model structure (depth={depth}):")
            print(format_model_tree(model, max_depth=depth))
        return model


@dataclass
class Pi05TokenizerConfig(_Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi05InferenceConfig(_Pi0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="/mlp_vepfs/share/yjh/checkpoints/dexbotic/libero_all_pi05/open16*8-0324/checkpoint-5000"
    )
    port: int = field(default=7891)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [6])
    action_dim: int = field(default=7)

    def _load_model(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Loading model from {self.model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        model = Pi05ForCausalLM.from_pretrained(
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
            ]
        )


@dataclass
class Pi05Exp(_Pi05Exp):
    model_config: Pi05ModelConfig = field(default_factory=Pi05ModelConfig)
    optimizer_config: Pi05OptimizerConfig = field(default_factory=Pi05OptimizerConfig)
    trainer_config: Pi05TrainerConfig = field(default_factory=Pi05TrainerConfig)
    data_config: Pi05DataConfig = field(default_factory=Pi05DataConfig)
    tokenizer_config: Pi05TokenizerConfig = field(default_factory=Pi05TokenizerConfig)
    inference_config: Pi05InferenceConfig = field(default_factory=Pi05InferenceConfig)

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = Pi05ComputeNormActionConfig()
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
        norm_config = Pi05ComputeNormActionConfig()
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
    exp = Pi05Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
