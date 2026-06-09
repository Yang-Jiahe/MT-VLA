import argparse
import hashlib
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import megfile
import numpy as np
from loguru import logger

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
from dexbotic.exp.pi0_exp import (
    Pi0ComputeNormActionConfig as _Pi0ComputeNormActionConfig,
)
from dexbotic.exp.pi0_exp import Pi0DataConfig as _Pi0DataConfig
from dexbotic.exp.pi0_exp import Pi0Exp as _Pi0Exp
from dexbotic.exp.pi0_exp import Pi0InferenceConfig as _Pi0InferenceConfig
from dexbotic.exp.pi0_exp import Pi0ModelConfig as _Pi0ModelConfig
from dexbotic.exp.pi0_exp import Pi0OptimizerConfig as _Pi0OptimizerConfig
from dexbotic.exp.pi0_exp import Pi0TokenizerConfig as _Pi0TokenizerConfig
from dexbotic.exp.pi0_exp import Pi0TrainerConfig as _Pi0TrainerConfig
from dexbotic.exp.pi0_exp import format_model_tree, _env_truthy
from dexbotic.model.pi0.pi0_arch import Pi0ForCausalLM
from dexbotic.model.pi0.moe_weight_loader import MoEWeightLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="train",
        choices=["train", "inference", "compute_norm_stats"],
    )
    args, _ = parser.parse_known_args()
    return args


class NormalizeKeys:
    def __init__(self, statistic_mapping: dict, keys: tuple[str, ...]):
        self.statistic_mapping = statistic_mapping
        self.keys = keys

    def __call__(self, episode_data_dict: dict, **kwargs) -> dict:
        for key in self.keys:
            if key not in episode_data_dict or key not in self.statistic_mapping:
                continue

            stats = self.statistic_mapping[key]
            if "mean" not in stats or "std" not in stats:
                continue

            data = episode_data_dict[key]
            mean = stats["mean"]
            std = stats["std"]
            if mean.shape[-1] < data.shape[-1]:
                mean = np.concatenate(
                    [mean, np.zeros(data.shape[-1] - mean.shape[-1])],
                    axis=-1,
                )
                std = np.concatenate(
                    [std, np.ones(data.shape[-1] - std.shape[-1])],
                    axis=-1,
                )
            elif mean.shape[-1] > data.shape[-1]:
                mean = mean[..., : data.shape[-1]]
                std = std[..., : data.shape[-1]]
            episode_data_dict[key] = ((data - mean) / (std + 1e-6)).astype(np.float32)
        return episode_data_dict


@dataclass
class MetaworldPi0OptimizerConfig(_Pi0OptimizerConfig):
    base_lr: float = field(default=2.5e-5)
    moe_lr: float = field(default=2.5e-5)
    router_lr: float = field(default=5e-5)
    moe_lora_lr: float = field(default=1e-4)
    moe_base_lr: float = field(default=2.5e-5)

    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
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
class MetaworldPi0TrainerConfig(_Pi0TrainerConfig):
    seed: int = field(default=42)
    bf16: bool = field(default=True)
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=5000)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=4)
    wandb_project: str = field(default="dexbotic_metaworld")
    model_max_length: int = field(default=48)
    output_dir: str = field(
        default=f"/mlp_vepfs/share/yjh/checkpoints/mtvla/metaworld_pi0/mt50-{datetime.now().strftime('%m%d_%H%M%S')}"
    )
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    task_aware_grad_accum: bool = field(default=True)
    pcgrad_on_shared: bool = field(default=True)
    pcgrad_grouping: str = field(default="prompt_hash")
    shared_expert_freeze_steps: int = field(default=20000)

@dataclass
class MetaworldPi0ComputeNormActionConfig(_Pi0ComputeNormActionConfig):
    trajectory_length: int = field(default=25)

    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    flatten=False,
                    padding_mode="last",
                    padding_action=False,
                ),
                ToList(),
            ]
        )
        return action_config

@dataclass
class MetaworldPi0ActionConfig(_Pi0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=25)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(
                    trajectory_length=self.trajectory_length,
                    flatten=False,
                    padding_mode="last",
                    padding_action=False,
                ),
                ActionNorm(statistic_mapping=statistic_mapping),
                LoadMultiModal(return_masks=True),
                ToList(),
            ]
        )
        return action_config


@dataclass
class MetaworldPi0DataConfig(_Pi0DataConfig):
    dataset_name: str = field(default="metaworld_mt50")
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
    action_config: MetaworldPi0ActionConfig = field(
        default_factory=MetaworldPi0ActionConfig
    )
    report_task_prompts: bool = field(default=True)
    task_id_from_prompt: bool = field(default=False)
    task_prompt_max_items: int = field(default=50)

@dataclass
class MetaworldPi0ModelConfig(_Pi0ModelConfig):
    model_name_or_path: str = field(default="/dexmal-fa-yjh-data/checkpoints/Dexbotic-PI0")
    chunk_size: int = field(default=25)
    use_moe: bool = field(default=True)
    moe_type: str = field(default="moe-lora")
    num_experts: int = field(default=4)
    moe_top_k: int = field(default=1)
    use_general_expert: bool = field(default=True)
    general_expert_weight: float = field(default=0.5)
    moe_lora_rank: int = field(default=4)
    moe_lora_alpha: Optional[float] = field(default=4)
    moe_load_balancing_weight: float = field(default=0.005)
    moe_stable: bool = field(default=True)
    moe_weight_path: Optional[str] = field(default="/mlp_vepfs/share/yjh/checkpoints/dexbotic/metaworld_pi0/checkpoint-30000/model-00002-of-00002.safetensors")
    moe_noise_std: float = field(default=0.0)
    moe_gating_init_std: float = field(default=0.006)
    print_model_structure: bool = field(default=True)
    model_structure_depth: int = field(default=6)

    task_token_enable: bool = field(default=True)
    task_token_use_task_id: bool = field(default=False)
    task_token_from_text: bool = field(default=True)
    task_token_pooling: str = field(default="mean")
    task_token_num_tasks: Optional[int] = field(default=40)

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
        if self.moe_stable:
            os.environ["DEXBOTIC_MOE_STABLE"] = "1"
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
        print(
            "[pi0] MoE config:",
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
                "[pi0] MoE-Lora config:",
                {
                    "moe_lora_rank": self.moe_lora_rank,
                    "moe_lora_alpha": self.moe_lora_alpha,
                },
            )
        config = Pi0ForCausalLM.config_class.from_pretrained(
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

        model = Pi0ForCausalLM.from_pretrained(self.model_name_or_path, config=config)
        model.config.chunk_size = self.chunk_size
        model.model.config.chunk_size = self.chunk_size

        if self.use_moe and self.moe_weight_path is not None:
            print(
                "[pi0] Loading MoE weights from",
                self.moe_weight_path,
            )
            loader = MoEWeightLoader(
                params_path=self.moe_weight_path,
                noise_std=self.moe_noise_std,
                gating_init_std=self.moe_gating_init_std,
            )
            loader.load(model, strict=False)
        if moe_type_norm == "moe-lora":
            reset_count = model.reset_moe_lora_params()
            if reset_count:
                print(f"[pi0] Reset MoE-LoRA params for {reset_count} layers")
        if self.print_model_structure or _env_truthy("DEXBOTIC_PRINT_MODEL"):
            depth = int(os.getenv("DEXBOTIC_MODEL_DEPTH", self.model_structure_depth))
            print(f"[pi0] Model structure (depth={depth}):")
            print(format_model_tree(model, max_depth=depth))
        return model


@dataclass
class MetaworldPi0TokenizerConfig(_Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class MetaworldPi0InferenceConfig(_Pi0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default=f"/mlp_vepfs/share/yjh/checkpoints/mtvla/metaworld_pi0/mt50-0603_005821/checkpoint-10000"
    )
    port: int = field(default=7893)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="/mlp_vepfs/share/yjh/debug_data/metaworld")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [3])
    action_dim: int = field(default=4)
    chunk_size: int = field(default=25)
    clip_action: bool = field(default=True)
    discrete_gripper: bool = field(default=False)

    def _load_model(self) -> None:
        super()._load_model()
        self.model.config.chunk_size = self.chunk_size
        self.model.model.config.chunk_size = self.chunk_size
        self.input_transform = Pipeline(
            [
                PadState(ndim=self.model.model.config.action_dim, axis=-1),
                NormalizeKeys(statistic_mapping=self.norm_stats, keys=("state",)),
                ToTensor(),
            ]
        )
        self.output_transform = Pipeline(
            [
                ToNumpy(),
                ActionDenorm(statistic_mapping=self.norm_stats, strict=False),
            ]
        )

    def _get_response(
        self,
        text: str | list[str],
        images: list[str],
        states: Optional[str | list[str]] = None,
        batch_size: int = 1,
    ) -> list[list[float]]:
        raw_output = super()._get_response(
            text=text,
            images=images,
            states=states,
            batch_size=batch_size,
        )
        actions = np.asarray(raw_output, dtype=np.float32)[..., : self.action_dim]
        actions = actions[: self.chunk_size]
        if self.clip_action:
            actions = np.clip(actions, -1.0, 1.0)
        if self.discrete_gripper and actions.shape[-1] >= 4:
            actions[..., 3] = np.where(actions[..., 3] > 0.0, 1.0, -1.0)
        return actions.tolist()


@dataclass
class MetaworldPi0Exp(_Pi0Exp):
    model_config: MetaworldPi0ModelConfig = field(
        default_factory=MetaworldPi0ModelConfig
    )
    optimizer_config: MetaworldPi0OptimizerConfig = field(
        default_factory=MetaworldPi0OptimizerConfig
    )
    trainer_config: MetaworldPi0TrainerConfig = field(
        default_factory=MetaworldPi0TrainerConfig
    )
    data_config: MetaworldPi0DataConfig = field(default_factory=MetaworldPi0DataConfig)
    tokenizer_config: MetaworldPi0TokenizerConfig = field(
        default_factory=MetaworldPi0TokenizerConfig
    )
    inference_config: MetaworldPi0InferenceConfig = field(
        default_factory=MetaworldPi0InferenceConfig
    )

    def inference(self) -> None:
        self.inference_config.run()

    def compute_norm_stats(self) -> None:
        self.data_config.action_config = MetaworldPi0ComputeNormActionConfig()
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
        action_config = self.data_config.action_config
        norm_config = MetaworldPi0ComputeNormActionConfig()
        save_name = hashlib.md5(self.data_config.dataset_name.encode()).hexdigest()[:8]
        norm_config.norm_save_path = os.path.join(
            os.path.dirname(norm_config.norm_save_path), save_name
        )
        norm_file_path = os.path.join(norm_config.norm_save_path, "norm_stats.json")
        if self.local_rank == 0 and not megfile.smart_exists(norm_file_path):
            logger.info("Auto-computing MetaWorld norm stats on rank0")
            self.compute_norm_stats()
        else:
            while not megfile.smart_exists(norm_file_path):
                time.sleep(5)
                print(
                    f"Waiting for norm stats: {norm_file_path} to be computed on rank{self.local_rank}"
                )
        action_config.statistic_mapping = norm_file_path
        self.data_config.action_config = action_config
        if self.local_rank == 0:
            print(
                f"Action config after auto compute norm: {self.data_config.action_config}"
            )


if __name__ == "__main__":
    args = parse_args()
    exp = MetaworldPi0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
