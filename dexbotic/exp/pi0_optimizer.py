from __future__ import annotations

from dataclasses import dataclass, field
import os

from loguru import logger
import torch
from transformers.trainer_pt_utils import get_parameter_names

try:
    from transformers.trainer import ALL_LAYERNORM_LAYERS
except ImportError:
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS

from dexbotic.exp.base_exp import OptimizerConfig
from dexbotic.model.pi0.pi0_arch import Pi0Model


@dataclass
class Pi0OptimizerConfig(OptimizerConfig):
    base_lr: float = field(default=2.5e-5)
    moe_lr: float = field(default=2.5e-5)
    router_lr: float = field(default=5e-5)
    moe_lora_lr: float = field(default=2.5e-5)
    moe_base_lr: float = field(default=2.5e-5)
    shared_expert_lr_scale: float | None = field(default=0.2)

    weight_decay: float = field(default=1e-6)
    moe_weight_decay: float = field(default=1e-6)
    router_weight_decay: float = field(default=1e-6)
    moe_lora_weight_decay: float = field(default=1e-6)
    moe_base_weight_decay: float = field(default=1e-6)

    adam_beta2: float = field(default=0.95)
    warmup_steps: int = field(default=1000)
    moe_warmup_steps: int = field(default=1000)
    router_warmup_steps: int = field(default=100)
    moe_lora_warmup_steps: int = field(default=1000)

    # Schedule metadata (kept for parity with AdaMoE JAX config).
    decay_steps: int = field(default=90_000)
    moe_decay_steps: int = field(default=90_000)
    router_decay_steps: int = field(default=90_000)
    moe_lora_decay_steps: int = field(default=90_000)
    decay_lr: float = field(default=1e-6)
    moe_decay_lr: float = field(default=1e-6)
    router_decay_lr: float = field(default=1e-6)
    moe_lora_decay_lr: float = field(default=1e-6)

    def _get_optimizer_grouped_parameters(self, model: Pi0Model) -> list:
        """Build parameter groups with MoE/router splits and decay handling."""
        decay_params_name = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_params_name = {name for name in decay_params_name if "bias" not in name}

        mm_projector_params_name = set()
        mm_vision_params_name = set()
        action_head_params_name = set()

        if self.mm_projector_lr is not None:
            logger.info("Using mm_projector_lr: {}", self.mm_projector_lr)
            mm_projector_params_name = {
                name
                for name, _ in model.named_parameters()
                if model.mm_projector_prefix in name
            }
        if self.mm_vision_lr is not None:
            logger.info("Using mm_vision_lr: {}", self.mm_vision_lr)
            mm_vision_params_name = {
                name for name, _ in model.named_parameters() if model.mm_vision_prefix in name
            }
        if self.action_head_lr is not None:
            logger.info("Using action_head_lr: {}", self.action_head_lr)
            action_head_params_name = {
                name
                for name, _ in model.named_parameters()
                if model.action_head_prefix in name
            }

        def is_decay(name: str) -> bool:
            return name in decay_params_name

        def is_router(name: str) -> bool:
            return "moe" in name and "gating" in name

        def is_moe_lora(name: str) -> bool:
            return "moe" in name and "lora_" in name

        def is_moe_base(name: str) -> bool:
            if "action_expert" not in name:
                return False
            if ".mlp." not in name:
                return False
            if "moe" in name:
                return False
            return any(
                token in name
                for token in (".gate_proj.", ".up_proj.", ".down_proj.")
            )

        def is_moe(name: str) -> bool:
            return "moe" in name and "lora_" not in name

        def is_mm_group(name: str) -> bool:
            return (
                name in mm_projector_params_name
                or name in mm_vision_params_name
                or name in action_head_params_name
            )

        base_decay = []
        base_no_decay = []
        moe_decay = []
        moe_no_decay = []
        router_decay = []
        router_no_decay = []
        lora_decay = []
        lora_no_decay = []
        moe_base_decay = []
        moe_base_no_decay = []
        base_names = []
        moe_names = []
        router_names = []
        lora_names = []
        moe_base_names = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if is_mm_group(name):
                continue
            if is_moe_base(name):
                (moe_base_decay if is_decay(name) else moe_base_no_decay).append(param)
                moe_base_names.append(name)
            elif is_moe_lora(name):
                (lora_decay if is_decay(name) else lora_no_decay).append(param)
                lora_names.append(name)
            elif is_router(name):
                (router_decay if is_decay(name) else router_no_decay).append(param)
                router_names.append(name)
            elif is_moe(name):
                (moe_decay if is_decay(name) else moe_no_decay).append(param)
                moe_names.append(name)
            else:
                (base_decay if is_decay(name) else base_no_decay).append(param)
                base_names.append(name)

        local_rank = int(os.getenv("LOCAL_RANK", "0") or 0)
        if local_rank == 0:
            def _log_group(name: str, names: list[str], params: list[torch.Tensor]) -> None:
                if not names:
                    return
                total_elems = sum(p.numel() for p in params)
                sample = ", ".join(names[:3])
                suffix = " ..." if len(names) > 3 else ""
                logger.info(
                    "Pi0 optimizer group {}: {} params ({} elements) [{}{}]",
                    name,
                    len(names),
                    total_elems,
                    sample,
                    suffix,
                )

            _log_group("base", base_names, base_decay + base_no_decay)
            _log_group("moe", moe_names, moe_decay + moe_no_decay)
            _log_group("router", router_names, router_decay + router_no_decay)
            _log_group("moe_lora", lora_names, lora_decay + lora_no_decay)
            _log_group(
                "moe_base", moe_base_names, moe_base_decay + moe_base_no_decay
            )

        parameters = []

        if self.mm_projector_lr is not None:
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in mm_projector_params_name and n in decay_params_name
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": self.mm_projector_lr,
                    "group_name": "mm_projector",
                    "group_role": "shared",
                }
            )
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad
                        and n in mm_projector_params_name
                        and n not in decay_params_name
                    ],
                    "weight_decay": 0.0,
                    "lr": self.mm_projector_lr,
                    "group_name": "mm_projector",
                    "group_role": "shared",
                }
            )

        if self.mm_vision_lr is not None:
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in mm_vision_params_name and n in decay_params_name
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": self.mm_vision_lr,
                    "group_name": "mm_vision",
                    "group_role": "shared",
                }
            )
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in mm_vision_params_name and n not in decay_params_name
                    ],
                    "weight_decay": 0.0,
                    "lr": self.mm_vision_lr,
                    "group_name": "mm_vision",
                    "group_role": "shared",
                }
            )

        if self.action_head_lr is not None:
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad and n in action_head_params_name and n in decay_params_name
                    ],
                    "weight_decay": self.weight_decay,
                    "lr": self.action_head_lr,
                    "group_name": "action_head",
                    "group_role": "shared",
                }
            )
            parameters.append(
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if p.requires_grad
                        and n in action_head_params_name
                        and n not in decay_params_name
                    ],
                    "weight_decay": 0.0,
                    "lr": self.action_head_lr,
                    "group_name": "action_head",
                    "group_role": "shared",
                }
            )

        if base_decay:
            parameters.append(
                {
                    "params": base_decay,
                    "weight_decay": self.weight_decay,
                    "lr": self.base_lr,
                    "group_name": "base",
                    "group_role": "shared",
                }
            )
        if base_no_decay:
            parameters.append(
                {
                    "params": base_no_decay,
                    "weight_decay": 0.0,
                    "lr": self.base_lr,
                    "group_name": "base",
                    "group_role": "shared",
                }
            )
        if moe_decay:
            parameters.append(
                {
                    "params": moe_decay,
                    "weight_decay": self.moe_weight_decay,
                    "lr": self.moe_lr,
                    "group_name": "moe",
                    "group_role": "shared",
                }
            )
        if moe_no_decay:
            parameters.append(
                {
                    "params": moe_no_decay,
                    "weight_decay": 0.0,
                    "lr": self.moe_lr,
                    "group_name": "moe",
                    "group_role": "shared",
                }
            )
        shared_lr = self.moe_base_lr
        if self.shared_expert_lr_scale is not None:
            shared_lr = self.base_lr * float(self.shared_expert_lr_scale)
        if moe_base_decay:
            parameters.append(
                {
                    "params": moe_base_decay,
                    "weight_decay": self.moe_base_weight_decay,
                    "lr": shared_lr,
                    "group_name": "moe_base",
                    "group_role": "shared",
                }
            )
        if moe_base_no_decay:
            parameters.append(
                {
                    "params": moe_base_no_decay,
                    "weight_decay": 0.0,
                    "lr": shared_lr,
                    "group_name": "moe_base",
                    "group_role": "shared",
                }
            )
        if lora_decay:
            parameters.append(
                {
                    "params": lora_decay,
                    "weight_decay": self.moe_lora_weight_decay,
                    "lr": self.moe_lora_lr,
                    "group_name": "moe_lora",
                    "group_role": "private",
                }
            )
        if lora_no_decay:
            parameters.append(
                {
                    "params": lora_no_decay,
                    "weight_decay": 0.0,
                    "lr": self.moe_lora_lr,
                    "group_name": "moe_lora",
                    "group_role": "private",
                }
            )
        if router_decay:
            parameters.append(
                {
                    "params": router_decay,
                    "weight_decay": self.router_weight_decay,
                    "lr": self.router_lr,
                    "group_name": "router",
                    "group_role": "shared",
                }
            )
        if router_no_decay:
            parameters.append(
                {
                    "params": router_no_decay,
                    "weight_decay": 0.0,
                    "lr": self.router_lr,
                    "group_name": "router",
                    "group_role": "shared",
                }
            )

        return parameters
