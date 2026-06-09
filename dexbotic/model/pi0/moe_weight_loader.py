from __future__ import annotations

import dataclasses
import os

import safetensors.torch
import torch


@dataclasses.dataclass(frozen=True)
class MoEWeightLoader:
    params_path: str
    num_experts: int = 4
    noise_std: float = 0.01
    gating_init_std: float = 0.006

    def load(self, model: torch.nn.Module, *, strict: bool = True) -> None:
        print(f"[MoEWeightLoader] Start: {self.num_experts} experts")
        raw_state = self._load_raw_state(self.params_path)
        model_state = model.state_dict()
        loaded: dict[str, torch.Tensor] = {}

        for key, value in raw_state.items():
            if key not in model_state:
                continue
            if value.shape != model_state[key].shape:
                print(
                    f"Skipping {key} due to shape mismatch: {tuple(value.shape)} vs {tuple(model_state[key].shape)}"
                )
                continue
            loaded[key] = value.to(model_state[key].dtype)

        if self._has_moe_lora(model_state):
            self._load_moe_lora_base(loaded, raw_state, model_state)
        if any(".mlp.gate_proj.weight" in k for k in raw_state):
            print("Found action expert templates for replication")
            self._replicate_experts(loaded, model_state)
        else:
            print("No action expert weights found for replication")
        self._init_gating(loaded, model_state)

        missing_count = 0
        for key, value in model_state.items():
            if key not in loaded:
                loaded[key] = value
                missing_count += 1

        print(f"[MoEWeightLoader] Completed: {missing_count} params filled from reference")

        missing, unexpected = model.load_state_dict(loaded, strict=strict)
        if missing or unexpected:
            print(f"MoEWeightLoader missing={missing} unexpected={unexpected}")
        print("[MoEWeightLoader] Done.")

    def _load_raw_state(self, path: str) -> dict[str, torch.Tensor]:
        if os.path.isdir(path):
            safetensors_path = os.path.join(path, "model.safetensors")
            if os.path.exists(safetensors_path):
                return safetensors.torch.load_file(safetensors_path)
            bin_path = os.path.join(path, "pytorch_model.bin")
            if os.path.exists(bin_path):
                return torch.load(bin_path, map_location="cpu")
        return safetensors.torch.load_file(path)

    @staticmethod
    def _has_moe_lora(model_state: dict[str, torch.Tensor]) -> bool:
        return any("moe_ffn.lora_" in key for key in model_state)

    @staticmethod
    def _lookup_raw_key(raw_state: dict[str, torch.Tensor], key: str) -> str | None:
        if key in raw_state:
            return key
        if key.startswith("model."):
            alt_key = key[len("model.") :]
            if alt_key in raw_state:
                return alt_key
        else:
            alt_key = f"model.{key}"
            if alt_key in raw_state:
                return alt_key
        return None

    @staticmethod
    def _is_moe_lora_base_key(key: str) -> bool:
        if "action_expert" not in key or ".mlp." not in key:
            return False
        if "moe_ffn" in key:
            return False
        return any(
            token in key
            for token in (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
        )

    def _load_moe_lora_base(
        self,
        loaded: dict[str, torch.Tensor],
        raw_state: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> None:
        for key, ref_weight in model_state.items():
            if not self._is_moe_lora_base_key(key):
                continue
            if key in loaded:
                continue
            src_key = self._lookup_raw_key(raw_state, key)
            if src_key is None:
                continue
            value = raw_state[src_key]
            if value.shape != ref_weight.shape:
                print(
                    f"Skipping {src_key} due to shape mismatch: {tuple(value.shape)} vs {tuple(ref_weight.shape)}"
                )
                continue
            if not torch.isfinite(value).all().item():
                print(f"Skipping {src_key} due to non-finite values")
                continue
            loaded[key] = value.to(ref_weight.dtype)

    def _replicate_experts(
        self,
        loaded: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> None:
        generator = torch.Generator()
        generator.manual_seed(42)
        for key, ref_weight in model_state.items():
            if "moe_ffn.w_expert_hidden" in key:
                self._replicate_hidden(key, ref_weight, loaded, model_state, generator)
            elif "moe_ffn.w_expert_output" in key:
                self._replicate_output(key, ref_weight, loaded, model_state, generator)

    def _replicate_hidden(
        self,
        key: str,
        ref_weight: torch.Tensor,
        loaded: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> None:
        prefix = key.split(".moe_ffn.w_expert_hidden", 1)[0]
        gate_key = f"{prefix}.gate_proj.weight"
        up_key = f"{prefix}.up_proj.weight"
        gate_weight = loaded.get(gate_key)
        up_weight = loaded.get(up_key)
        if gate_weight is None or up_weight is None:
            print(f"Missing template weights for {key}")
            return

        template = torch.stack([gate_weight.T, up_weight.T], dim=0)
        template = template[:, None, :, :].expand_as(ref_weight)
        noise = torch.randn(
            ref_weight.shape,
            dtype=ref_weight.dtype,
            device=ref_weight.device,
            generator=generator,
        )
        noise = noise * self.noise_std
        loaded[key] = (template + noise).to(ref_weight.dtype)

    def _replicate_output(
        self,
        key: str,
        ref_weight: torch.Tensor,
        loaded: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
        generator: torch.Generator,
    ) -> None:
        prefix = key.split(".moe_ffn.w_expert_output", 1)[0]
        down_key = f"{prefix}.down_proj.weight"
        down_weight = loaded.get(down_key)
        if down_weight is None:
            print(f"Missing template weights for {key}")
            return

        template = down_weight.T[None, :, :].expand_as(ref_weight)
        noise = torch.randn(
            ref_weight.shape,
            dtype=ref_weight.dtype,
            device=ref_weight.device,
            generator=generator,
        )
        noise = noise * self.noise_std
        loaded[key] = (template + noise).to(ref_weight.dtype)

    def _init_gating(
        self,
        loaded: dict[str, torch.Tensor],
        model_state: dict[str, torch.Tensor],
    ) -> None:
        for key, ref_weight in model_state.items():
            if "moe_ffn.w_gating" not in key and "moe_ffn.w_gating_scale" not in key:
                continue
            if key in loaded:
                continue
            generator = torch.Generator(device=ref_weight.device)
            generator.manual_seed(0)
            noise = torch.randn(
                ref_weight.shape,
                dtype=ref_weight.dtype,
                device=ref_weight.device,
                generator=generator,
            )
            loaded[key] = noise * self.gating_init_std
