import os
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, DynamicCache, CONFIG_MAPPING
from transformers.models.gemma.modeling_gemma import (
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from dexbotic.model.dexbotic_arch import (
    ActionOutputForCausalLM,
    CausalLMOutputDexbotic,
    DexboticConfig,
    DexboticForCausalLM,
    DexboticVLMModel,
)
from dexbotic.model.pi0.moe_pytorch import MoEGemmaMLP


def make_attn_mask(input_mask: torch.BoolTensor, ar_mask: torch.BoolTensor):
    ar_mask = ar_mask.broadcast_to(input_mask.shape)
    cumsum = torch.cumsum(ar_mask, dim=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    # Mask keys only; masking queries can produce all -inf rows and NaNs in softmax.
    valid_mask = input_mask[:, None, :]
    attn_mask = torch.logical_and(attn_mask, valid_mask)
    # Ensure every query has at least one valid key to avoid all-masked rows.
    row_has_key = attn_mask.any(dim=-1, keepdim=True)
    attn_mask = torch.where(row_has_key, attn_mask, valid_mask)
    return attn_mask


def make_attn_mask_4d(attn_mask: torch.BoolTensor):
    attn_mask = torch.where(attn_mask, 0.0, -2.3819763e38)[:, None]
    return attn_mask


def posemb_sincos(
    position_ids: torch.LongTensor,
    dim: int,
    min_period: int,
    max_period: int,
):
    if dim % 2 != 0:
        raise ValueError("dim must be even for sincos position embeddings")

    fraction = torch.linspace(0.0, 1.0, dim // 2, dtype=torch.float64).to(
        position_ids.device
    )
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = position_ids[:, None].float() / period[None, :] * 2 * np.pi
    return torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1)


class Pi0Config(DexboticConfig):
    model_type = "dexbotic_pi0"
    vision_config: dict | str | None = None
    processor_config: str | None = None
    action_config: dict | str | None = None
    action_dim: Optional[int] = 32
    chunk_size: Optional[int] = 50
    task_token_enable: bool = True
    task_token_use_task_id: bool = True
    task_token_from_text: bool = False
    task_token_pooling: str = "mean"
    task_token_num_tasks: Optional[int] = None

    def __init__(self, *args, **kwargs):
        kwargs = dict(kwargs)
        if "top_k" in kwargs and "moe_top_k" not in kwargs:
            kwargs["moe_top_k"] = kwargs.pop("top_k")
        action_config = kwargs.get("action_config", None)
        if isinstance(action_config, dict):
            if "top_k" in action_config and "moe_top_k" not in action_config:
                action_config = dict(action_config)
                action_config["moe_top_k"] = action_config.pop("top_k")
                kwargs["action_config"] = action_config
        super().__init__(*args, **kwargs)
        vision_config = kwargs.pop("vision_config", None)
        if isinstance(vision_config, dict):
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](
                **vision_config
            )
        elif isinstance(vision_config, str):
            self.vision_config = AutoConfig.from_pretrained(vision_config)

        action_config = kwargs.get("action_config", None)
        if isinstance(action_config, dict):
            self.action_config = CONFIG_MAPPING[action_config["model_type"]](
                **action_config
            )
        elif isinstance(action_config, str):
            self.action_config = AutoConfig.from_pretrained(action_config)
        elif action_config is not None:
            self.action_config = action_config

        llm_config = kwargs.get("llm_config", None)
        if isinstance(llm_config, dict):
            self.llm_config = CONFIG_MAPPING[llm_config["model_type"]](**llm_config)
        elif isinstance(llm_config, str):
            self.llm_config = AutoConfig.from_pretrained(llm_config)

        action_cfg = getattr(self, "action_config", None)
        action_use_moe = getattr(action_cfg, "use_moe", False) if action_cfg else False
        action_moe_type = getattr(action_cfg, "moe_type", "adamoe") if action_cfg else "adamoe"
        action_num_experts = getattr(action_cfg, "num_experts", 4) if action_cfg else 4
        action_moe_top_k = (getattr(action_cfg, "moe_top_k", getattr(action_cfg, "top_k", 1))if action_cfg else 1)
        action_use_general = getattr(action_cfg, "use_general_expert", True) if action_cfg else True
        action_general_weight = getattr(action_cfg, "general_expert_weight", 0.5) if action_cfg else 0.5
        action_moe_lora_rank = (getattr(action_cfg, "moe_lora_rank", None) if action_cfg else None)
        action_moe_lora_alpha = (getattr(action_cfg, "moe_lora_alpha", None) if action_cfg else None)

        self.use_moe = bool(getattr(self, "use_moe", action_use_moe))
        self.moe_type = getattr(self, "moe_type", action_moe_type)
        self.num_experts = int(getattr(self, "num_experts", action_num_experts))
        self.moe_top_k = int(getattr(self, "moe_top_k", action_moe_top_k))
        self.use_general_expert = bool(getattr(self, "use_general_expert", action_use_general))
        self.general_expert_weight = float(getattr(self, "general_expert_weight", action_general_weight))
        self.moe_load_balancing_weight = float(getattr(self, "moe_load_balancing_weight", 0.005))
        
        moe_type_norm = str(self.moe_type).lower().replace("_", "-")
        if moe_type_norm == "moe-lora":
            lora_rank = getattr(self, "moe_lora_rank", None)
            if lora_rank is None:
                lora_rank = action_moe_lora_rank
            if lora_rank is None:
                lora_rank = 8
            self.moe_lora_rank = int(lora_rank)
            lora_alpha = getattr(self, "moe_lora_alpha", action_moe_lora_alpha)
            self.moe_lora_alpha = float(lora_alpha) if lora_alpha is not None else None

        if action_cfg is not None:
            self.action_config.use_moe = self.use_moe
            self.action_config.moe_type = self.moe_type
            self.action_config.num_experts = self.num_experts
            self.action_config.moe_top_k = self.moe_top_k
            self.action_config.use_general_expert = self.use_general_expert
            self.action_config.general_expert_weight = self.general_expert_weight
            if moe_type_norm == "moe-lora":
                self.action_config.moe_lora_rank = self.moe_lora_rank
                self.action_config.moe_lora_alpha = self.moe_lora_alpha

        self.task_token_enable = bool(getattr(self, "task_token_enable", True))
        self.task_token_use_task_id = bool(
            getattr(self, "task_token_use_task_id", True)
        )
        self.task_token_from_text = bool(
            getattr(self, "task_token_from_text", False)
        )
        self.task_token_pooling = str(
            getattr(self, "task_token_pooling", "mean")
        ).lower()
        if self.task_token_pooling not in {"mean", "first"}:
            self.task_token_pooling = "mean"
        num_tasks = getattr(self, "task_token_num_tasks", None)
        self.task_token_num_tasks = int(num_tasks) if num_tasks is not None else None


class Pi0Model(DexboticVLMModel):
    def __init__(self, config: Pi0Config):
        super().__init__(config)

        action_model_config = config.action_config
        self.action_expert = AutoModel.from_config(action_model_config)
        self.use_moe = bool(getattr(action_model_config, "use_moe", False))
        if self.use_moe:
            if not hasattr(self.action_expert, "layers"):
                raise ValueError("MoE requires Gemma-style action expert with layers.")
            for layer in self.action_expert.layers:
                layer.mlp = MoEGemmaMLP(action_model_config)
        self.state_proj = nn.Linear(config.action_dim, action_model_config.hidden_size)
        self.action_in_proj = nn.Linear(
            config.action_dim, action_model_config.hidden_size
        )
        self.action_time_mlp_in = nn.Linear(
            2 * action_model_config.hidden_size, action_model_config.hidden_size
        )
        self.action_time_activation = nn.SiLU()
        self.action_time_mlp_out = nn.Linear(
            action_model_config.hidden_size, action_model_config.hidden_size
        )
        self.action_out_proj = nn.Linear(
            action_model_config.hidden_size, config.action_dim
        )
        self.task_token_enable = bool(getattr(config, "task_token_enable", True))
        self.task_token_use_task_id = bool(
            getattr(config, "task_token_use_task_id", True)
        )
        self.task_token_from_text = bool(
            getattr(config, "task_token_from_text", False)
        )
        self.task_token_pooling = str(
            getattr(config, "task_token_pooling", "mean")
        ).lower()
        if self.task_token_pooling not in {"mean", "first"}:
            self.task_token_pooling = "mean"
        num_tasks = getattr(config, "task_token_num_tasks", None)
        if self.task_token_enable and self.task_token_use_task_id and num_tasks is not None:
            self.task_token_embed = nn.Embedding(
                int(num_tasks), action_model_config.hidden_size
            )
            nn.init.normal_(self.task_token_embed.weight, mean=0.0, std=0.02)
        else:
            self.task_token_embed = None
        llm_hidden = config.llm_config.hidden_size
        action_hidden = action_model_config.hidden_size
        if self.task_token_enable and self.task_token_from_text and llm_hidden != action_hidden:
            self.task_token_proj = nn.Linear(llm_hidden, action_hidden)
        else:
            self.task_token_proj = None
        torch.set_float32_matmul_precision("highest")

    def collect_moe_info(self):
        if not self.use_moe:
            return None

        infos = []
        for layer in self.action_expert.layers:
            layer_info = getattr(layer.mlp, "last_moe_info", None)
            if layer_info is not None:
                infos.append(layer_info)

        if not infos:
            return None

        return {
            "z_loss": torch.stack([info.z_loss for info in infos]).mean(),
            "load_balancing_loss": torch.stack(
                [info.load_balancing_loss for info in infos]
            ).mean(),
            "activation_square_sum": torch.stack(
                [info.activation_square_sum for info in infos]
            ).mean(),
        }

    def reset_moe_lora_params(self) -> int:
        if not self.use_moe:
            return 0
        layers = getattr(self.action_expert, "layers", None)
        if not layers:
            return 0
        reset_count = 0
        for layer in layers:
            mlp = getattr(layer, "mlp", None)
            moe_ffn = getattr(mlp, "moe_ffn", None) if mlp is not None else None
            if moe_ffn is None or not hasattr(moe_ffn, "reset_lora_params"):
                continue
            moe_ffn.reset_lora_params()
            reset_count += 1
        return reset_count


class Pi0ForCausalLM(DexboticForCausalLM, ActionOutputForCausalLM):
    config_class = Pi0Config

    def _real_init(self, config: Pi0Config):
        self.model = Pi0Model(config)
        self._moe_step = 0
        self.post_init()

    def reset_moe_lora_params(self) -> int:
        return self.model.reset_moe_lora_params()

    @staticmethod
    def _is_kernel_param(name: str, param: torch.Tensor) -> bool:
        if param.ndim <= 1:
            return False
        exclude = (
            "bias",
            "scale",
            "pos_embedding",
            "position_embedding",
            "input_embedding",
            "embed_tokens",
        )
        return not any(token in name for token in exclude)

    @staticmethod
    def _global_norm(params: list[torch.Tensor]) -> torch.Tensor:
        if not params:
            return torch.tensor(0.0)
        device = params[0].device
        total = torch.zeros((), device=device)
        for param in params:
            total += torch.sum(param.detach().float() ** 2)
        return torch.sqrt(total)

    def _collect_param_norms(self) -> dict[str, torch.Tensor]:
        total_params: list[torch.Tensor] = []
        moe_params: list[torch.Tensor] = []
        router_params: list[torch.Tensor] = []
        report_anomalies = os.getenv("DEXBOTIC_REPORT_PARAM_ANOMALIES", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }
        max_items = int(os.getenv("DEXBOTIC_ANOMALY_MAX_ITEMS", "10") or 10)
        max_abs_env = os.getenv("DEXBOTIC_ANOMALY_MAX_ABS", "").strip()
        max_abs = float(max_abs_env) if max_abs_env else None
        nonfinite_entries: list[tuple[str, int, float]] = []
        nonfinite_count = 0
        large_entries: list[tuple[str, float]] = []
        large_count = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if not self._is_kernel_param(name, param):
                continue
            total_params.append(param)
            if "moe" in name:
                moe_params.append(param)
                if "gating" in name:
                    router_params.append(param)
            if report_anomalies:
                data = param.detach()
                finite_mask = torch.isfinite(data)
                if not finite_mask.all().item():
                    nonfinite_count += 1
                    if len(nonfinite_entries) < max_items:
                        nonfinite = int((~finite_mask).sum().item())
                        if finite_mask.any().item():
                            max_abs_finite = float(
                                data[finite_mask].abs().max().item()
                            )
                        else:
                            max_abs_finite = float("inf")
                        nonfinite_entries.append(
                            (name, nonfinite, max_abs_finite)
                        )
                    continue
                if max_abs is not None:
                    max_val = float(data.abs().max().item())
                    if max_val > max_abs:
                        large_count += 1
                        if len(large_entries) < max_items:
                            large_entries.append((name, max_val))
        if report_anomalies and nonfinite_count:
            if not getattr(self, "_reported_param_anomalies", False):
                print(
                    "[pi0] Non-finite params detected: "
                    f"{nonfinite_count} (showing {len(nonfinite_entries)})"
                )
                for name, nonfinite, max_abs_finite in nonfinite_entries:
                    print(
                        "[pi0] non-finite param:",
                        name,
                        "nonfinite=",
                        nonfinite,
                        "max_abs_finite=",
                        f"{max_abs_finite:.4g}",
                    )
        if report_anomalies and max_abs is not None and large_count:
            if not getattr(self, "_reported_param_anomalies", False):
                print(
                    "[pi0] Large params detected: "
                    f"{large_count} (max_abs>{max_abs:.4g}, showing {len(large_entries)})"
                )
                for name, max_val in large_entries:
                    print(
                        "[pi0] large param:",
                        name,
                        "max_abs=",
                        f"{max_val:.4g}",
                    )
        if report_anomalies and (nonfinite_count or large_count):
            self._reported_param_anomalies = True
        return {
            "moe_norm": self._global_norm(moe_params),
            "router_norm": self._global_norm(router_params),
            "total_param_norm": self._global_norm(total_params),
        }

    def _inner_forward_mot(
        self,
        module_list: List[nn.Module],
        input_embeds_list: List[torch.Tensor],
        mask: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        past_key_values: Optional[DynamicCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        update_cache: bool = True,
    ):
        force_attn_fp32 = os.getenv("DEXBOTIC_FORCE_ATTENTION_FP32", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

        all_hidden_states = (input_embeds_list,) if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for layer_idx, layers in enumerate(
            zip(*[module.layers for module in module_list])
        ):
            query_list, key_list, value_list = [], [], []
            seq_len_list = []
            for module_idx, (layer, input_embeds) in enumerate(
                zip(layers, input_embeds_list)
            ):
                if input_embeds is None:
                    seq_len_list.append(0)
                else:
                    prenorm_embeds = layer.input_layernorm(input_embeds)
                    batch_size, seq_len, _ = prenorm_embeds.shape
                    seq_len_list.append(seq_len)

                    query = (
                        layer.self_attn.q_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    key = (
                        layer.self_attn.k_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    value = (
                        layer.self_attn.v_proj(prenorm_embeds)
                        .view(batch_size, seq_len, -1, layer.self_attn.head_dim)
                        .transpose(1, 2)
                    )
                    query_list.append(query)
                    key_list.append(key)
                    value_list.append(value)

            query_states = torch.cat(query_list, dim=2)
            key_states = torch.cat(key_list, dim=2)
            value_states = torch.cat(value_list, dim=2)
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, *position_embeddings
            )

            if past_key_values is not None:
                if update_cache:
                    key_states, value_states = past_key_values.update(
                        key_states, value_states, layer_idx
                    )
                else:
                    key_states = torch.cat(
                        [past_key_values.key_cache[layer_idx], key_states], dim=-2
                    )
                    value_states = torch.cat(
                        [past_key_values.value_cache[layer_idx], value_states], dim=-2
                    )

            if force_attn_fp32:
                q_states = query_states.float()
                k_states = key_states.float()
                v_states = value_states.float()
                attn_output, attn_weights = eager_attention_forward(
                    layers[0].self_attn,
                    q_states,
                    k_states,
                    v_states,
                    mask,
                    layers[0].self_attn.scaling,
                )
                attn_output = attn_output.to(query_states.dtype)
            else:
                attn_output, attn_weights = eager_attention_forward(
                    layers[0].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    mask,
                    layers[0].self_attn.scaling,
                )
            if output_attentions:
                all_self_attns += (attn_weights,)

            attn_output = attn_output.view(batch_size, sum(seq_len_list), -1)
            layer_embeds_list = []
            start_idx = 0
            for module_idx, (layer, input_embeds) in enumerate(
                zip(layers, input_embeds_list)
            ):
                seq_len = seq_len_list[module_idx]
                if seq_len == 0:
                    layer_embeds_list.append(None)
                    continue
                attn_embeds = attn_output[:, start_idx : start_idx + seq_len, :]
                start_idx += seq_len

                attn_embeds = layer.self_attn.o_proj(attn_embeds)
                residual_attn_embeds = input_embeds + attn_embeds
                postnorm_embeds = layer.post_attention_layernorm(residual_attn_embeds)
                mlp_embeds = layer.mlp(postnorm_embeds)
                residual_mlp_embeds = residual_attn_embeds + mlp_embeds
                layer_embeds_list.append(residual_mlp_embeds)

            input_embeds_list = layer_embeds_list

        decoder_embeds_list = []
        for module_idx, (module, input_embeds) in enumerate(
            zip(module_list, input_embeds_list)
        ):
            if input_embeds is not None:
                input_embeds = module.norm(input_embeds)
            decoder_embeds_list.append(input_embeds)

        if output_hidden_states:
            all_hidden_states += (decoder_embeds_list,)
        return decoder_embeds_list, past_key_values, all_hidden_states, all_self_attns

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.mm_vision_module(images)
        image_features = self.model.mm_projector_module(image_features)
        return image_features

    def embed_prefix(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        return_input_tokens: bool = False,
    ):
        input_mask = []
        ar_mask = []
        tokens = []
        input_tokens = None

        images = images.transpose(0, 1)
        image_masks = image_masks.transpose(0, 1)
        for image, image_mask in zip(images, image_masks):
            image_tokens = self.encode_images(image)
            tokens.append(image_tokens)
            image_mask = image_mask.unsqueeze(1).expand(
                image.shape[0], image_tokens.shape[1]
            )
            input_mask.append(image_mask)
            ar_mask += [False] * image_tokens.shape[1]

        if input_ids is not None:
            input_tokens = (
                self.model.llm.embed_tokens(input_ids)
                * self.model.config.llm_config.hidden_size**0.5
            )
            input_mask.append(attention_mask)
            ar_mask += [False] * input_tokens.shape[1]
            tokens.append(input_tokens)

        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        if return_input_tokens:
            return tokens, input_mask, ar_mask, input_tokens
        return tokens, input_mask, ar_mask

    def _pool_task_embedding(
        self,
        input_tokens: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor],
    ) -> Optional[torch.FloatTensor]:
        if input_tokens is None:
            return None
        pooling = str(getattr(self.model, "task_token_pooling", "mean")).lower()
        if pooling == "first":
            return input_tokens[:, 0]
        if attention_mask is None:
            return input_tokens.mean(dim=1)
        mask = attention_mask.to(dtype=input_tokens.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (input_tokens * mask).sum(dim=1) / denom

    def _compute_task_token(
        self,
        input_tokens: Optional[torch.FloatTensor],
        attention_mask: Optional[torch.Tensor],
        task_ids: Optional[torch.LongTensor],
    ) -> Optional[torch.FloatTensor]:
        if not getattr(self.model, "task_token_enable", True):
            return None
        task_token = None
        task_id_embed = getattr(self.model, "task_token_embed", None)
        if task_ids is not None and task_id_embed is not None:
            if task_ids.dim() > 1:
                task_ids = task_ids[:, 0]
            task_token = task_id_embed(task_ids)
        if task_token is None and getattr(self.model, "task_token_from_text", False):
            task_token = self._pool_task_embedding(input_tokens, attention_mask)
            if task_token is not None:
                task_token_proj = getattr(self.model, "task_token_proj", None)
                if task_token_proj is not None:
                    task_token = task_token_proj(task_token)
        return task_token

    def embed_suffix(
        self,
        states: Optional[torch.FloatTensor] = None,
        noisy_actions: Optional[torch.FloatTensor] = None,
        time: Optional[torch.FloatTensor] = None,
        task_token: Optional[torch.FloatTensor] = None,
        cast_state_to_float: bool = False,
    ):
        input_mask = []
        ar_mask = []
        tokens = []

        if task_token is not None:
            task_token = task_token.unsqueeze(1)
            tokens.append(task_token)
            input_mask.append(
                torch.ones(
                    (task_token.shape[0], 1),
                    device=task_token.device,
                    dtype=torch.bool,
                )
            )
            ar_mask.append(True)

        if cast_state_to_float:
            state_token = self.model.state_proj(states.float()).unsqueeze(1)
        else:
            state_token = self.model.state_proj(states).unsqueeze(1)

        tokens.append(state_token)
        input_mask.append(
            torch.ones((states.shape[0], 1), device=states.device, dtype=torch.bool)
        )
        ar_mask.append(True)

        time_emb = posemb_sincos(
            time,
            self.model.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
        )
        time_emb = time_emb.unsqueeze(1)
        time_tokens = time_emb.expand(-1, self.model.config.chunk_size, -1)
        action_tokens = self.model.action_in_proj(noisy_actions)
        action_time_tokens = torch.cat(
            [action_tokens, time_tokens.to(action_tokens.dtype)], dim=-1
        )
        action_time_tokens = self.model.action_time_mlp_in(action_time_tokens)
        action_time_tokens = self.model.action_time_activation(action_time_tokens)
        action_time_tokens = self.model.action_time_mlp_out(action_time_tokens)
        tokens.append(action_time_tokens)
        input_mask.append(
            torch.ones(
                action_time_tokens.shape[:2],
                device=action_time_tokens.device,
                dtype=torch.bool,
            )
        )
        ar_mask += [True] + ([False] * (self.model.config.chunk_size - 1))
        tokens = torch.cat(tokens, dim=1)
        input_mask = torch.cat(input_mask, dim=1)
        ar_mask = torch.tensor(ar_mask, device=tokens.device)
        return tokens, input_mask, ar_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        actions: Optional[torch.FloatTensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        repeated_diffusion_steps: int = 4,
        image_masks: Optional[torch.BoolTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        cast_state_to_float: bool = False,
        **kwargs,
    ) -> CausalLMOutputDexbotic:
        batch_shape = actions.shape[:1]
        noise = torch.normal(
            mean=torch.zeros_like(actions),
            std=torch.ones_like(actions),
        ).to(
            device=actions.device,
            dtype=actions.dtype,
        )
        time = (
            torch.distributions.Beta(1.5, 1)
            .sample(batch_shape)
            .to(device=actions.device, dtype=actions.dtype)
            * 0.999
            + 0.001
        )
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, prefix_ar_mask, input_tokens = self.embed_prefix(
            input_ids,
            attention_mask,
            images,
            image_masks,
            return_input_tokens=True,
        )
        task_token = self._compute_task_token(input_tokens, attention_mask, task_ids)
        suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
            states, x_t, time, task_token=task_token, cast_state_to_float=False
        )
        input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
        ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        attn_mask = make_attn_mask_4d(attn_mask)
        positions = torch.cumsum(input_mask, dim=1) - 1
        position_embeddings = self.model.llm.rotary_emb(prefix_tokens, positions)

        (prefix_out, suffix_out), past_key_values, hidden_states, attentions = (
            self._inner_forward_mot(
                [self.model.llm, self.model.action_expert],
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                cache_position=positions,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
            )
        )

        # with torch.amp.autocast(device_type="cuda", dtype=torch.float32):
        v_t = self.model.action_out_proj(suffix_out[:, -self.model.config.chunk_size :])
        flow_matching_loss_per_sample = F.mse_loss(v_t, u_t, reduction="none").mean(
            dim=(1, 2)
        )
        flow_matching_loss = flow_matching_loss_per_sample.mean()
        loss = flow_matching_loss
        moe_info = self.model.collect_moe_info()
        if moe_info is not None and "load_balancing_loss" in moe_info:
            lb_weight = kwargs.get("moe_load_balancing_weight")
            if lb_weight is None:
                step = kwargs.get("moe_step")
                if step is None and self.training:
                    step = self._moe_step
                if step is not None:
                    lb_weight = 0.01 if step <= 500 else 0.005
                else:
                    lb_weight = getattr(self.config, "moe_load_balancing_weight", 0.005)
            loss = loss + moe_info["load_balancing_loss"].mean() * lb_weight
            if self.training:
                self._moe_step += 1

        if output_hidden_states:
            hidden_states += (v_t,)

        outputs = CausalLMOutputDexbotic(
            loss=loss,
            logits=v_t,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions,
        )
        if self.training:
            param_norms = self._collect_param_norms()
            outputs["moe_norm"] = param_norms["moe_norm"]
            outputs["router_norm"] = param_norms["router_norm"]
            outputs["total_param_norm"] = param_norms["total_param_norm"]
        outputs["flow_matching_loss"] = flow_matching_loss
        outputs["flow_matching_loss_per_sample"] = flow_matching_loss_per_sample
        if moe_info is not None:
            outputs["moe_z_loss"] = moe_info["z_loss"]
            outputs["moe_load_balancing_loss"] = moe_info["load_balancing_loss"]
            outputs["moe_activation_square_sum"] = moe_info["activation_square_sum"]
        return outputs

    @torch.no_grad()
    def inference_action(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        states: Optional[torch.FloatTensor] = None,
        images: Optional[torch.FloatTensor] = None,
        image_masks: Optional[torch.BoolTensor] = None,
        diffusion_steps: int = 10,
        task_ids: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        batch_size = states.shape[0]

        dt = -1.0 / diffusion_steps
        noise = torch.normal(
            0,
            1,
            size=(batch_size, self.model.config.chunk_size, self.config.action_dim),
            device=states.device,
        )
        time = torch.tensor(1.0, device=states.device)

        prefix_tokens, prefix_mask, prefix_ar_mask, input_tokens = self.embed_prefix(
            input_ids,
            attention_mask,
            images,
            image_masks,
            return_input_tokens=True,
        )
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_attn_mask = make_attn_mask_4d(prefix_attn_mask)
        positions = torch.cumsum(prefix_mask, dim=1) - 1
        position_embeddings = self.model.llm.rotary_emb(prefix_tokens, positions)
        _, kv_cache, _, _ = self._inner_forward_mot(
            [self.model.llm, self.model.action_expert],
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            position_embeddings=position_embeddings,
            past_key_values=DynamicCache(),
            cache_position=positions,
            output_hidden_states=False,
            output_attentions=False,
        )

        def step(x_t, time):
            task_token = self._compute_task_token(input_tokens, attention_mask, task_ids)
            suffix_tokens, suffix_mask, suffix_ar_mask = self.embed_suffix(
                states, x_t, time.broadcast_to(batch_size), task_token=task_token, cast_state_to_float=True
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = prefix_mask.unsqueeze(1).repeat(
                1, suffix_tokens.shape[1], 1
            )
            full_attn_mask = torch.cat([prefix_attn_mask, suffix_attn_mask], dim=-1)
            full_attn_mask = make_attn_mask_4d(full_attn_mask)
            assert full_attn_mask.shape == (
                batch_size,
                1,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            full_positions = (
                prefix_mask.sum(axis=-1).unsqueeze(-1)
                + torch.cumsum(suffix_mask, dim=-1)
                - 1
            )
            full_position_embeddings = self.model.llm.rotary_emb(
                suffix_tokens, full_positions
            )
            (prefix_out, suffix_out), _, _, _ = self._inner_forward_mot(
                [self.model.llm, self.model.action_expert],
                [None, suffix_tokens],
                mask=full_attn_mask,
                position_embeddings=full_position_embeddings,
                past_key_values=kv_cache,
                cache_position=torch.cat(
                    [positions, torch.cumsum(suffix_mask, dim=1) - 1], dim=1
                ),
                output_hidden_states=False,
                output_attentions=False,
                update_cache=False,
            )
            assert prefix_out is None
            v_t = self.model.action_out_proj(
                suffix_out[:, -self.model.config.chunk_size :]
            )
            return x_t + v_t * dt, time + dt

        while time > -dt / 2:
            noise, time = step(noise, time)

        return noise

    def process_images(self, images):
        vision_tower = self.model.mm_vision_module
        image_processor = vision_tower.image_processor
        image_aspect_ratio = getattr(self.config, "image_aspect_ratio", "pad")
        new_images = []
        if image_aspect_ratio == "pad":
            for image in images:
                image = self.expand2square(
                    image, tuple(int(x * 255) for x in [0, 0, 0])
                )
                image = image_processor.preprocess(image, return_tensors="pt")[
                    "pixel_values"
                ][0]
                new_images.append(image)
        else:
            return image_processor(images, return_tensors="pt")["pixel_values"]
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images
