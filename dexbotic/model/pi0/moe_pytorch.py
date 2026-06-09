from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812
from transformers.models.gemma import modeling_gemma


@dataclass(frozen=True)
class MoEInfo:
    z_loss: torch.Tensor
    load_balancing_loss: torch.Tensor
    activation_square_sum: torch.Tensor


class BaseMoEFeedForward(nn.Module):
    def __init__(
        self,
        *,
        expert_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self._init_std = init_std
        self._sanitized = False

        self.w_gating = nn.Parameter(torch.empty(expert_dim, num_experts))
        self.w_expert_hidden = nn.Parameter(
            torch.empty(2, num_experts, expert_dim, hidden_dim)
        )
        self.w_expert_output = nn.Parameter(
            torch.empty(num_experts, hidden_dim, expert_dim)
        )

        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.w_gating, mean=0.0, std=init_std)
        nn.init.normal_(self.w_expert_hidden, mean=0.0, std=init_std)
        nn.init.normal_(self.w_expert_output, mean=0.0, std=init_std)

    def _maybe_sanitize_params(self) -> None:
        sanitize_always = os.getenv("DEXBOTIC_MOE_SANITIZE_ALWAYS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if self._sanitized and not sanitize_always:
            return
        if os.getenv("DEXBOTIC_MOE_SANITIZE", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            return
        with torch.no_grad():
            params = [self.w_gating, self.w_expert_hidden, self.w_expert_output]
            if hasattr(self, "w_gating_scale"):
                params.append(self.w_gating_scale)
            needs_reset = any(not torch.isfinite(p).all().item() for p in params)
            if needs_reset:
                self.reset_parameters(self._init_std)
                if hasattr(self, "w_gating_scale"):
                    nn.init.normal_(self.w_gating_scale, mean=0.0, std=0.02)
            else:
                for param in params:
                    if not torch.isfinite(param).all().item():
                        param.data = torch.nan_to_num(
                            param.data, nan=0.0, posinf=0.0, neginf=0.0
                        )
                if hasattr(self, "w_gating_scale"):
                    gating_scale = self.w_gating_scale
                    max_abs = float(gating_scale.detach().abs().max().item())
                    clamp_abs = float(
                        os.getenv("DEXBOTIC_GATING_SCALE_CLAMP", "10.0")
                    )
                    gating_scale.data = torch.nan_to_num(
                        gating_scale.data, nan=0.0, posinf=0.0, neginf=0.0
                    )
                    gating_scale.data.clamp_(-clamp_abs, clamp_abs)
        if not sanitize_always:
            self._sanitized = True

    @staticmethod
    def _z_loss(logits: torch.Tensor, *, lambda_z: float = 1e-4) -> torch.Tensor:
        logz = torch.logsumexp(logits, dim=-1)
        return lambda_z * torch.mean(logz**2)

    def gating_topk(self, x: torch.Tensor, gate_scale_fn=None):
        self._maybe_sanitize_params()
        force_moe_fp32 = os.getenv("DEXBOTIC_FORCE_MOE_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = os.getenv("DEXBOTIC_MOE_STABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = stable_moe or force_moe_fp32
        batch, seq, dim = x.shape
        x_flat = x.reshape(-1, dim)
        x_mat = x_flat.float() if force_moe_fp32 else x_flat

        logits = torch.matmul(x_mat, self.w_gating.to(x_mat.dtype))
        if stable_moe:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        z_loss = self._z_loss(logits)

        gate_probs = F.softmax(logits, dim=-1)
        if stable_moe:
            gate_probs = torch.nan_to_num(gate_probs, nan=0.0, posinf=0.0, neginf=0.0)
        mean_gate_score = gate_probs.mean(dim=0)

        effective_top_k = min(self.top_k, self.num_experts)
        effective_top_k = max(effective_top_k, 1)
        top_k_values, top_k_indices = torch.topk(
            gate_probs, effective_top_k, dim=-1
        )
        if stable_moe:
            top_k_indices = top_k_indices.clamp(0, self.num_experts - 1)
        total_tokens = batch * seq
        one_hot_topk = F.one_hot(top_k_indices, self.num_experts).float()
        expert_token_counts = one_hot_topk.sum(dim=(0, 1))
        activation_ratio = expert_token_counts / float(total_tokens)
        activation_square_sum = torch.sum(activation_ratio**2)
        load_balancing_loss = torch.mean(activation_ratio * mean_gate_score)

        moe_info = MoEInfo(
            z_loss=z_loss,
            load_balancing_loss=load_balancing_loss,
            activation_square_sum=activation_square_sum,
        )

        if gate_scale_fn is not None:
            gate_scales = gate_scale_fn(x_mat, gate_probs, top_k_indices)
        else:
            gate_scales = gate_probs
        if stable_moe:
            gate_scales = torch.nan_to_num(
                gate_scales, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(min=-10.0, max=10.0)

        return x_mat, top_k_indices, gate_scales, moe_info

    def forward_expert(
        self,
        x_flat: torch.Tensor,
        top_k_indices: torch.Tensor,
        gate_scales: torch.Tensor,
    ) -> torch.Tensor:
        stable_moe = os.getenv("DEXBOTIC_MOE_STABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        force_moe_fp32 = os.getenv("DEXBOTIC_FORCE_MOE_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = stable_moe or force_moe_fp32
        w_hidden = self.w_expert_hidden.to(x_flat.dtype)
        w_output = self.w_expert_output.to(x_flat.dtype)

        ff_gate = torch.einsum("bf, efh -> beh", x_flat, w_hidden[0])
        if stable_moe:
            ff_gate = torch.nan_to_num(ff_gate, nan=0.0, posinf=0.0, neginf=0.0)
        gate_value = F.gelu(ff_gate)
        ff1 = torch.einsum("bf, efh -> beh", x_flat, w_hidden[1])
        if stable_moe:
            ff1 = torch.nan_to_num(ff1, nan=0.0, posinf=0.0, neginf=0.0)
        expert_hidden = gate_value * ff1
        expert_output = torch.einsum("beh, ehd -> bed", expert_hidden, w_output)
        if stable_moe:
            expert_output = torch.nan_to_num(
                expert_output, nan=0.0, posinf=0.0, neginf=0.0
            )

        selected_outputs = torch.gather(
            expert_output,
            dim=1,
            index=top_k_indices[..., None].expand(-1, -1, self.expert_dim),
        )
        if gate_scales.shape == top_k_indices.shape:
            selected_scales = gate_scales
        else:
            selected_scales = torch.gather(gate_scales, dim=1, index=top_k_indices)
        selected_scales = selected_scales.to(selected_outputs.dtype)

        weighted_outputs = torch.sum(
            selected_outputs * selected_scales[..., None], dim=1
        )
        if stable_moe:
            weighted_outputs = torch.nan_to_num(
                weighted_outputs, nan=0.0, posinf=0.0, neginf=0.0
            )
        return weighted_outputs


class VanillaMoEFeedForward(BaseMoEFeedForward):
    def forward(self, x: torch.Tensor):
        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(x)
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape_as(x)
        return output, moe_info


class AdaMoEFeedForward(BaseMoEFeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_gating_scale = nn.Parameter(
            torch.empty(self.expert_dim, self.num_experts)
        )
        nn.init.normal_(self.w_gating_scale, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        def gate_scale_fn(x_flat, gate_probs, _top_k_indices):
            scale = torch.matmul(x_flat, self.w_gating_scale.to(x_flat.dtype))
            return gate_probs + scale

        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(
            x, gate_scale_fn=gate_scale_fn
        )
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape_as(x)
        return output, moe_info


class CSMoEFeedForward(BaseMoEFeedForward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_gating_scale = nn.Parameter(
            torch.empty(self.expert_dim + self.num_experts, self.num_experts)
        )
        nn.init.normal_(self.w_gating_scale, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor):
        def gate_scale_fn(x_flat, gate_probs, _top_k_indices):
            scale_input = torch.cat([x_flat, gate_probs], dim=-1)
            return F.softmax(
                torch.matmul(scale_input, self.w_gating_scale.to(x_flat.dtype)), dim=-1
            )

        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(
            x, gate_scale_fn=gate_scale_fn
        )
        weighted_outputs = self.forward_expert(x_flat, top_k_indices, gate_scales)
        output = weighted_outputs.reshape_as(x)
        return output, moe_info


class MoELoRAFeedForward(nn.Module):
    def __init__(
        self,
        *,
        expert_dim: int,
        hidden_dim: int,
        num_experts: int = 4,
        top_k: int = 1,
        init_std: float = 0.02,
        lora_rank: int = 8,
        lora_alpha: float | None = None,
    ):
        super().__init__()
        self.expert_dim = expert_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self._init_std = init_std
        self._sanitized = False

        self.lora_rank = max(int(lora_rank), 1)
        if lora_alpha is None:
            lora_alpha = float(self.lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_scale = self.lora_alpha / float(self.lora_rank)

        self.w_gating = nn.Parameter(torch.empty(expert_dim, num_experts))

        self.lora_gate_a = nn.Parameter(
            torch.empty(num_experts, expert_dim, self.lora_rank)
        )
        self.lora_gate_b = nn.Parameter(
            torch.empty(num_experts, self.lora_rank, hidden_dim)
        )
        self.lora_up_a = nn.Parameter(
            torch.empty(num_experts, expert_dim, self.lora_rank)
        )
        self.lora_up_b = nn.Parameter(
            torch.empty(num_experts, self.lora_rank, hidden_dim)
        )
        self.lora_down_a = nn.Parameter(
            torch.empty(num_experts, hidden_dim, self.lora_rank)
        )
        self.lora_down_b = nn.Parameter(
            torch.empty(num_experts, self.lora_rank, expert_dim)
        )

        self.reset_parameters(init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.normal_(self.w_gating, mean=0.0, std=init_std)
        nn.init.normal_(self.lora_gate_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_gate_b)
        nn.init.normal_(self.lora_up_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_up_b)
        nn.init.normal_(self.lora_down_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_down_b)

    def reset_lora_params(self, init_std: float | None = None) -> None:
        if init_std is None:
            init_std = self._init_std
        nn.init.normal_(self.lora_gate_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_gate_b)
        nn.init.normal_(self.lora_up_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_up_b)
        nn.init.normal_(self.lora_down_a, mean=0.0, std=init_std)
        nn.init.zeros_(self.lora_down_b)
        self._sanitized = False

    def _maybe_sanitize_params(self) -> None:
        sanitize_always = os.getenv("DEXBOTIC_MOE_SANITIZE_ALWAYS", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        if self._sanitized and not sanitize_always:
            return
        if os.getenv("DEXBOTIC_MOE_SANITIZE", "0").strip().lower() not in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }:
            return
        with torch.no_grad():
            params = [
                self.w_gating,
                self.lora_gate_a,
                self.lora_gate_b,
                self.lora_up_a,
                self.lora_up_b,
                self.lora_down_a,
                self.lora_down_b,
            ]
            max_abs_env = os.getenv("DEXBOTIC_LORA_MAX_ABS", "").strip().lower()
            if max_abs_env in {"0", "none", "off", "false"}:
                max_abs = None
            elif max_abs_env:
                max_abs = float(max_abs_env)
            else:
                max_abs = 1e3
            needs_reset = any(not torch.isfinite(p).all().item() for p in params)
            if not needs_reset and max_abs is not None:
                max_val = max(float(p.detach().abs().max().item()) for p in params)
                if max_val > max_abs:
                    needs_reset = True
                    if not getattr(self, "_reported_large", False):
                        print(
                            "[pi0] Resetting MoE-LoRA params due to large values:",
                            f"max_abs={max_val:.4g}",
                        )
                        self._reported_large = True
            if needs_reset:
                self.reset_parameters(self._init_std)
            else:
                for param in params:
                    if not torch.isfinite(param).all().item():
                        param.data = torch.nan_to_num(
                            param.data, nan=0.0, posinf=0.0, neginf=0.0
                        )
        if not sanitize_always:
            self._sanitized = True

    @staticmethod
    def _z_loss(logits: torch.Tensor, *, lambda_z: float = 1e-4) -> torch.Tensor:
        logz = torch.logsumexp(logits, dim=-1)
        return lambda_z * torch.mean(logz**2)

    def gating_topk(self, x: torch.Tensor):
        self._maybe_sanitize_params()
        force_moe_fp32 = os.getenv("DEXBOTIC_FORCE_MOE_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = os.getenv("DEXBOTIC_MOE_STABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = stable_moe or force_moe_fp32
        batch, seq, dim = x.shape
        x_flat = x.reshape(-1, dim)
        x_mat = x_flat.float() if force_moe_fp32 else x_flat

        logits = torch.matmul(x_mat, self.w_gating.to(x_mat.dtype))
        if stable_moe:
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        z_loss = self._z_loss(logits)

        gate_probs = F.softmax(logits, dim=-1)
        if stable_moe:
            gate_probs = torch.nan_to_num(gate_probs, nan=0.0, posinf=0.0, neginf=0.0)
        mean_gate_score = gate_probs.mean(dim=0)

        effective_top_k = min(self.top_k, self.num_experts)
        effective_top_k = max(effective_top_k, 1)
        top_k_values, top_k_indices = torch.topk(
            gate_probs, effective_top_k, dim=-1
        )
        if stable_moe:
            top_k_indices = top_k_indices.clamp(0, self.num_experts - 1)
        total_tokens = batch * seq
        one_hot_topk = F.one_hot(top_k_indices, self.num_experts).float()
        expert_token_counts = one_hot_topk.sum(dim=(0, 1))
        activation_ratio = expert_token_counts / float(total_tokens)
        activation_square_sum = torch.sum(activation_ratio**2)
        load_balancing_loss = torch.mean(activation_ratio * mean_gate_score)

        moe_info = MoEInfo(
            z_loss=z_loss,
            load_balancing_loss=load_balancing_loss,
            activation_square_sum=activation_square_sum,
        )

        gate_scales = gate_probs
        if stable_moe:
            gate_scales = torch.nan_to_num(
                gate_scales, nan=0.0, posinf=0.0, neginf=0.0
            ).clamp(min=-10.0, max=10.0)

        return x_mat, top_k_indices, gate_scales, moe_info

    def _apply_lora(
        self, x_flat: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor
    ) -> torch.Tensor:
        x_a = torch.einsum("bf, efr -> ber", x_flat, lora_a.to(x_flat.dtype))
        out = torch.einsum("ber, erh -> beh", x_a, lora_b.to(x_flat.dtype))
        return out * self.lora_scale

    def _apply_lora_hidden(
        self, x_hidden: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor
    ) -> torch.Tensor:
        x_a = torch.einsum("beh, ehr -> ber", x_hidden, lora_a.to(x_hidden.dtype))
        out = torch.einsum("ber, erd -> bed", x_a, lora_b.to(x_hidden.dtype))
        return out * self.lora_scale

    def forward(
        self,
        x: torch.Tensor,
        base_gate_weight: torch.Tensor,
        base_up_weight: torch.Tensor,
        base_down_weight: torch.Tensor,
    ):
        x_flat, top_k_indices, gate_scales, moe_info = self.gating_topk(x)
        stable_moe = os.getenv("DEXBOTIC_MOE_STABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        force_moe_fp32 = os.getenv("DEXBOTIC_FORCE_MOE_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        stable_moe = stable_moe or force_moe_fp32

        base_gate = F.linear(x_flat, base_gate_weight.to(x_flat.dtype))
        base_up = F.linear(x_flat, base_up_weight.to(x_flat.dtype))
        if stable_moe:
            base_gate = torch.nan_to_num(base_gate, nan=0.0, posinf=0.0, neginf=0.0)
            base_up = torch.nan_to_num(base_up, nan=0.0, posinf=0.0, neginf=0.0)

        gate_out = base_gate[:, None, :] + self._apply_lora(
            x_flat, self.lora_gate_a, self.lora_gate_b
        )
        up_out = base_up[:, None, :] + self._apply_lora(
            x_flat, self.lora_up_a, self.lora_up_b
        )
        if stable_moe:
            gate_out = torch.nan_to_num(gate_out, nan=0.0, posinf=0.0, neginf=0.0)
            up_out = torch.nan_to_num(up_out, nan=0.0, posinf=0.0, neginf=0.0)

        expert_hidden = F.gelu(gate_out) * up_out
        expert_output = F.linear(
            expert_hidden, base_down_weight.to(expert_hidden.dtype)
        )
        expert_output = expert_output + self._apply_lora_hidden(
            expert_hidden, self.lora_down_a, self.lora_down_b
        )
        if stable_moe:
            expert_output = torch.nan_to_num(
                expert_output, nan=0.0, posinf=0.0, neginf=0.0
            )

        selected_outputs = torch.gather(
            expert_output,
            dim=1,
            index=top_k_indices[..., None].expand(-1, -1, self.expert_dim),
        )
        if gate_scales.shape == top_k_indices.shape:
            selected_scales = gate_scales
        else:
            selected_scales = torch.gather(gate_scales, dim=1, index=top_k_indices)
        selected_scales = selected_scales.to(selected_outputs.dtype)

        weighted_outputs = torch.sum(
            selected_outputs * selected_scales[..., None], dim=1
        )
        if stable_moe:
            weighted_outputs = torch.nan_to_num(
                weighted_outputs, nan=0.0, posinf=0.0, neginf=0.0
            )

        output = weighted_outputs.reshape_as(x)
        return output, moe_info


def _normalize_moe_type(value) -> str:
    if isinstance(value, str):
        return value.lower().replace("_", "-")
    if hasattr(value, "value"):
        return str(value.value).lower().replace("_", "-")
    return str(value).lower().replace("_", "-")


class MoEGemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = modeling_gemma.ACT2FN[config.hidden_act]

        self.use_moe = bool(getattr(config, "use_moe", False))
        self.moe_type = _normalize_moe_type(getattr(config, "moe_type", "adamoe"))
        self.num_experts = int(getattr(config, "num_experts", 4))
        self.moe_top_k = int(getattr(config, "moe_top_k", getattr(config, "top_k", 1)))
        self.use_general_expert = bool(getattr(config, "use_general_expert", True))
        self.general_expert_weight = 0.5

        if self.use_moe:
            moe_kwargs = {
                "expert_dim": config.hidden_size,
                "hidden_dim": config.intermediate_size,
                "num_experts": self.num_experts,
                "top_k": self.moe_top_k,
                "init_std": getattr(config, "initializer_range", 0.02),
            }
            if self.moe_type == "vanilla":
                self.moe_ffn = VanillaMoEFeedForward(**moe_kwargs)
            elif self.moe_type == "csmoe":
                self.moe_ffn = CSMoEFeedForward(**moe_kwargs)
            elif self.moe_type == "moe-lora":
                self.moe_ffn = MoELoRAFeedForward(
                    **moe_kwargs,
                    lora_rank=int(getattr(config, "moe_lora_rank", 8)),
                    lora_alpha=getattr(config, "moe_lora_alpha", None),
                )
            else:
                self.moe_ffn = AdaMoEFeedForward(**moe_kwargs)
        else:
            self.moe_ffn = None

        self.last_moe_info: MoEInfo | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        force_fp32 = os.getenv("DEXBOTIC_FORCE_MLP_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        force_moe_fp32 = os.getenv("DEXBOTIC_FORCE_MOE_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        disable_moe = os.getenv("DEXBOTIC_DISABLE_MOE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        def general_ffn(x_in):
            if not (force_fp32 or force_moe_fp32):
                return self.down_proj(
                    self.act_fn(self.gate_proj(x_in)) * self.up_proj(x_in)
                )
            gate_out = F.linear(x_in, self.gate_proj.weight.float())
            up_out = F.linear(x_in, self.up_proj.weight.float())
            hidden = self.act_fn(gate_out) * up_out
            return F.linear(hidden, self.down_proj.weight.float())

        if disable_moe or not self.use_moe or self.moe_ffn is None:
            self.last_moe_info = None
            if force_fp32 or force_moe_fp32:
                return general_ffn(x.float()).to(x.dtype)
            return general_ffn(x)

        moe_input = x.float() if (force_fp32 or force_moe_fp32) else x
        if isinstance(self.moe_ffn, MoELoRAFeedForward):
            moe_out, moe_info = self.moe_ffn(
                moe_input,
                self.gate_proj.weight,
                self.up_proj.weight,
                self.down_proj.weight,
            )
        else:
            moe_out, moe_info = self.moe_ffn(moe_input)
        if force_fp32 or force_moe_fp32:
            moe_out = moe_out.to(x.dtype)
        self.last_moe_info = moe_info

        if self.use_general_expert:
            general_out = general_ffn(moe_input)
            if force_fp32 or force_moe_fp32:
                general_out = general_out.to(x.dtype)
            weight = 0.5
            return general_out * weight + moe_out * (1.0 - weight)

        return moe_out
