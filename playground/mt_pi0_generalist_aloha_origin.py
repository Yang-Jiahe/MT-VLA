import argparse
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from dexbotic.data.dataset.transform.action import (
    ActionNorm,
    AddTrajectory,
    DeltaAction,
    PadAction,
    PadState,
)
from dexbotic.data.dataset.transform.common import Pipeline, ToDict, ToList, ToNumpy
from dexbotic.data.dataset.transform.multimodal import LoadMultiModal
from dexbotic.exp.pi0_exp import Pi0InferenceConfig as _Pi0InferenceConfig
from dexbotic.exp.pi0_exp import Pi0Exp as _Pi0Exp
from dexbotic.exp.pi0_exp import Pi0DataConfig as _Pi0DataConfig
from dexbotic.exp.pi0_exp import Pi0ModelConfig as _Pi0ModelConfig
from dexbotic.exp.pi0_exp import Pi0OptimizerConfig as _Pi0OptimizerConfig
from dexbotic.exp.pi0_exp import Pi0TrainerConfig as _Pi0TrainerConfig
from dexbotic.exp.pi0_exp import (
    Pi0ComputeNormActionConfig as _Pi0ComputeNormActionConfig,
)
from dexbotic.exp.pi0_exp import Pi0ActionConfig as _Pi0ActionConfig
from dexbotic.exp.pi0_exp import Pi0TokenizerConfig as _Pi0TokenizerConfig
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
    args, unknown = parser.parse_known_args()
    return args


@dataclass
class Pi0OptimizerConfig(_Pi0OptimizerConfig):
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
class Pi0TrainerConfig(_Pi0TrainerConfig):
    num_train_steps: int = field(default=30000)
    save_steps: int = field(default=5000)
    save_total_limit: Optional[int] = field(default=5)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=4)
    model_max_length: int = field(default=48)
    moe_type = os.getenv("MOE_TYPE", "")
    output_dir: str = field(
        default=f"/mlp_vepfs/share/yjh/checkpoints/rc_table30_cvpr_generalist/mtvla_pi0/{moe_type}/all2k_origin_{datetime.now().strftime('%m%d')}"
    )
    wandb_project: str = field(default='rc_table30_cvpr_generalist')
    lr_scheduler_type: str = field(default="cosine_with_min_lr")
    lr_scheduler_kwargs: dict = field(default_factory=lambda: {"min_lr_rate": 0.1})
    task_aware_grad_accum: bool = field(default=True)
    pcgrad_on_shared: bool = field(default=True)
    pcgrad_grouping: str = field(default="prompt_hash")
    shared_expert_freeze_steps: int = field(default=20000)

    feature_vis_enable: bool = field(default=False)
    feature_vis_steps: int = field(default=100)
    feature_vis_num_samples: int = field(default=800)
    feature_vis_max_scan: int = field(default=8000)
    feature_vis_output_dir: str = field(default="")
    feature_vis_probe_batch_size: int = field(default=8)
    feature_vis_probe_use_images: bool = field(default=True)

    bf16: bool = field(default=True) # False True
    # gradient_checkpointing: bool = field(default=False)    
    # dataloader_num_workers: int = field(default=2)
    # logging_steps: int = field(default=10)
    # use_jax_style_warmup: bool = field(default=True)
    # seed: int = field(default=42)
    # # ===== 禁用 DeepSpeed 以使 raw_backward 生效 =====
    # deepspeed: str = field(default=None)
    # # ===== 消融：使用 loss.backward() 替代 accelerator.backward() =====
    # use_raw_backward: bool = field(default=True)

class Pi0ComputeNormActionConfig(_Pi0ComputeNormActionConfig):
    def build_action_process_func(self) -> Pipeline:
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
                PadState(ndim=32, axis=-1),
                PadAction(ndim=32, axis=-1),
                AddTrajectory(trajectory_length=50, flatten=False, padding_mode="last"),
                DeltaAction(enable=True),
                ToList(),
            ]
        )

        return action_config


@dataclass
class Pi0ActionConfig(_Pi0ActionConfig):
    statistic_mapping: str = field(default=None)
    trajectory_length: int = field(default=50)

    def build_action_process_func(self) -> Pipeline:
        statistic_mapping = self._read_norm_stats(self.statistic_mapping)
        action_config = Pipeline(
            [
                ToDict(),
                ToNumpy(),
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
class Pi0DataConfig(_Pi0DataConfig):
    dataset_name: str = field(
        default="+".join([
            "table30_cvpr_generalist_lint_roller_remove_dirt_200",
            "table30_cvpr_generalist_pack_the_items_200",
            "table30_cvpr_generalist_pack_the_toothbrush_holder_200",
            "table30_cvpr_generalist_paint_jam_200",
            "table30_cvpr_generalist_put_the_books_back_200",
            "table30_cvpr_generalist_put_the_pencil_case_into_the_schoolbag_200",
            "table30_cvpr_generalist_scoop_with_a_small_spoon_200",
            "table30_cvpr_generalist_stamp_positioning_200",
            "table30_cvpr_generalist_wipe_the_blackboard_200",
            "table30_cvpr_generalist_wrap_with_a_soft_cloth_200",
        ])
    )
    # Equal weights for all datasets
    dataset_weights: list[float] = field(
        default_factory=lambda: [1.0] * 11
    )
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
    action_config: Pi0ActionConfig = field(default_factory=Pi0ActionConfig)
    report_task_prompts: bool = field(default=True)
    task_id_from_prompt: bool = field(default=False)
    task_prompt_max_items: int = field(default=50)

@dataclass
class Pi0ModelConfig(_Pi0ModelConfig):
    model_name_or_path: str = field(default="/dexmal-fa-yjh-data/checkpoints/Dexbotic-PI0")
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
    moe_weight_path: Optional[str] = field(default='/dexmal-fa-yjh-data/checkpoints/Dexbotic-PI0')
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
class Pi0TokenizerConfig(_Pi0TokenizerConfig):
    use_fast_tokenizer: bool = field(default=False)


@dataclass
class Pi0InferenceConfig(_Pi0InferenceConfig):
    model_name_or_path: Optional[str] = field(
        default="/mlp_vepfs/share/yjh/checkpoints/mtvla7/libero/moe-lora/8_prompt_b001_0215")
    port: int = field(default=7895)
    save_image: bool = field(default=False)
    save_image_dir: str = field(default="./debug_data")
    norm_stats: Optional[dict] = field(default=None)
    num_images: int = field(default=3)
    non_delta_mask: list[int] = field(default_factory=lambda: [12, 13])
    action_dim: int = field(default=14)


@dataclass
class Pi0Exp(_Pi0Exp):
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


if __name__ == "__main__":
    args = parse_args()
    exp = Pi0Exp()
    if args.task == "train":
        exp.train()
    elif args.task == "inference":
        exp.inference()
    elif args.task == "compute_norm_stats":
        exp.compute_norm_stats()
