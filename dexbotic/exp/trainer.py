import os
import hashlib
import json
from typing import TYPE_CHECKING, Optional
import shutil

import torch
import numpy as np
import transformers
from loguru import logger
from easydict import EasyDict
from transformers import Trainer, TrainingArguments

from dexbotic.exp.utils import get_mm_adapter_state_maybe_zero_3
from dexbotic.model.dexbotic_arch import DexboticVLMModel
from dexbotic.model.pi0.pi0_arch import make_attn_mask, make_attn_mask_4d

if TYPE_CHECKING:
    from dexbotic.exp.base_exp import BaseExp


class DexboticTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.exp_config: BaseExp = kwargs.pop("exp_config")
        training_args = self._link_exp_config()
        super().__init__(*args, args=training_args, **kwargs)
        self.loss_cache = {}
        self._task_grad_buffer = {}
        self._task_count_buffer = {}
        self._shared_param_cache = None
        self._shared_expert_param_cache = None
        self._last_feature_vis_step = -1
        self._feature_vis_ref_indices: Optional[list[int]] = None
        self._feature_vis_ref_keys: Optional[list[str]] = None
        self._feature_vis_label_map: dict[str, int] = {}
        self._feature_vis_pca_mean: Optional[torch.Tensor] = None
        self._feature_vis_pca_basis: Optional[torch.Tensor] = None

    @staticmethod
    def _dist_enabled() -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def _dist_all_true(self, flag: bool) -> bool:
        if not self._dist_enabled():
            return flag
        device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        value = torch.tensor([1 if flag else 0], device=device, dtype=torch.int32)
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MIN)
        return bool(int(value.item()))

    def create_optimizer(self) -> torch.optim.Optimizer:
        opt_model: DexboticVLMModel = self.model

        if self.optimizer is None:
            optimizer_grouped_parameters = self.exp_config.optimizer_config._get_optimizer_grouped_parameters(
                opt_model)

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _task_aware_enabled(self) -> bool:
        return bool(
            getattr(self.exp_config.trainer_config, "task_aware_grad_accum", False)
        )

    def _pcgrad_enabled(self) -> bool:
        return bool(getattr(self.exp_config.trainer_config, "pcgrad_on_shared", False))

    def _pcgrad_grouping(self) -> str:
        grouping = str(
            getattr(self.exp_config.trainer_config, "pcgrad_grouping", "prompt_hash")
        ).lower()
        if grouping not in {"prompt_hash", "task_id"}:
            grouping = "prompt_hash"
        return grouping

    @staticmethod
    def _build_prompt_hash_keys(
        input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> list[str]:
        ids = input_ids.detach().to("cpu")
        mask = (
            attention_mask.detach().to("cpu").bool()
            if attention_mask is not None
            else None
        )
        keys: list[str] = []
        for idx in range(ids.shape[0]):
            valid = ids[idx][mask[idx]] if mask is not None else ids[idx]
            if valid.numel() == 0:
                keys.append("empty")
                continue
            digest = hashlib.blake2b(
                valid.contiguous().numpy().tobytes(), digest_size=8
            ).hexdigest()
            keys.append(digest)
        return keys

    def _build_group_keys(
        self, inputs: dict[str, torch.Tensor]
    ) -> Optional[list[str]]:
        grouping = self._pcgrad_grouping()
        if grouping == "task_id":
            task_ids = inputs.get("task_ids")
            if task_ids is None:
                return None
            task_ids = task_ids.view(-1)
            return [f"task_{int(v.item())}" for v in task_ids]
        input_ids = inputs.get("input_ids")
        if input_ids is None:
            return None
        attention_mask = inputs.get("attention_mask")
        return self._build_prompt_hash_keys(input_ids, attention_mask)

    def _shared_param_group_names(self) -> list[str]:
        return list(
            getattr(
                self.exp_config.trainer_config,
                "shared_param_group_names",
                [],
            )
        )

    def _shared_expert_group_names(self) -> list[str]:
        return list(
            getattr(
                self.exp_config.trainer_config,
                "shared_expert_group_names",
                [],
            )
        )

    def _shared_expert_freeze_steps(self) -> int:
        return int(
            getattr(self.exp_config.trainer_config, "shared_expert_freeze_steps", 0)
        )

    def _get_params_by_group(self, group_names: list[str]) -> list[torch.Tensor]:
        if not group_names:
            return []
        params = []
        for group in self.optimizer.param_groups:
            if group.get("group_name") in group_names:
                params.extend(group["params"])
        return params

    def _get_shared_params(self) -> list[torch.Tensor]:
        if self._shared_param_cache is None:
            group_names = self._shared_param_group_names()
            params = self._get_params_by_group(group_names)
            self._shared_param_cache = params
        return self._shared_param_cache

    def _get_shared_expert_params(self) -> list[torch.Tensor]:
        if self._shared_expert_param_cache is None:
            group_names = self._shared_expert_group_names()
            params = self._get_params_by_group(group_names)
            self._shared_expert_param_cache = params
        return self._shared_expert_param_cache

    def _gradient_conflict_stats_enabled(self) -> bool:
        return bool(
            getattr(
                self.exp_config.trainer_config,
                "gradient_conflict_stats_enable",
                True,
            )
        )

    def _gradient_conflict_stats_interval(self) -> int:
        return max(
            1,
            int(
                getattr(
                    self.exp_config.trainer_config,
                    "gradient_conflict_stats_steps",
                    1,
                )
            ),
        )

    def _gradient_conflict_stats_dir(self) -> str:
        out_dir = getattr(
            self.exp_config.trainer_config,
            "gradient_conflict_stats_output_dir",
            "",
        ).strip()
        if out_dir:
            return out_dir
        return os.path.join(self.args.output_dir, "gradient_conflict")

    @staticmethod
    def _grad_pair_cosine(
        grad_a: list[torch.Tensor | None],
        grad_b: list[torch.Tensor | None],
    ) -> Optional[float]:
        dot = None
        norm_a = None
        norm_b = None
        for ga, gb in zip(grad_a, grad_b):
            if ga is None or gb is None:
                continue
            ga_f = ga.detach().float()
            gb_f = gb.detach().float()
            dot_value = torch.sum(ga_f * gb_f)
            norm_a_value = torch.sum(ga_f * ga_f)
            norm_b_value = torch.sum(gb_f * gb_f)
            dot = dot_value if dot is None else dot + dot_value
            norm_a = norm_a_value if norm_a is None else norm_a + norm_a_value
            norm_b = norm_b_value if norm_b is None else norm_b + norm_b_value
        if dot is None or norm_a is None or norm_b is None:
            return None
        denom = torch.sqrt(norm_a.clamp(min=1e-30)) * torch.sqrt(
            norm_b.clamp(min=1e-30)
        )
        if denom.item() <= 0.0:
            return None
        cosine = dot / denom
        if not torch.isfinite(cosine).item():
            return None
        return float(cosine.item())

    def _compute_gradient_conflict_stats(
        self, task_grads: list[list[torch.Tensor | None]]
    ) -> dict[str, float]:
        num_groups = len(task_grads)
        cosines: list[float] = []
        for i in range(num_groups):
            for j in range(i + 1, num_groups):
                cosine = self._grad_pair_cosine(task_grads[i], task_grads[j])
                if cosine is not None:
                    cosines.append(cosine)
        if not cosines:
            return {
                "num_pairs": 0.0,
                "neg_ratio": 0.0,
                "conflict_intensity": 0.0,
                "mean_cosine": 0.0,
                "min_cosine": 0.0,
                "max_cosine": 0.0,
            }
        values = np.asarray(cosines, dtype=np.float64)
        neg_values = values[values < 0.0]
        return {
            "num_pairs": float(values.size),
            "neg_ratio": float(neg_values.size / float(values.size)),
            "conflict_intensity": float((-neg_values).mean()) if neg_values.size else 0.0,
            "mean_cosine": float(values.mean()),
            "min_cosine": float(values.min()),
            "max_cosine": float(values.max()),
        }

    def _record_gradient_conflict_stats(
        self,
        pre_stats: dict[str, float],
        post_stats: dict[str, float],
        *,
        num_groups: int,
    ) -> None:
        if not self._gradient_conflict_stats_enabled():
            return
        step = int(self.state.global_step)
        interval = self._gradient_conflict_stats_interval()
        if step > 0 and step % interval != 0:
            return

        record = {
            "step": step,
            "num_instruction_groups": float(num_groups),
            "tgd_pre_num_pairs": pre_stats["num_pairs"],
            "tgd_pre_neg_ratio": pre_stats["neg_ratio"],
            "tgd_pre_conflict_intensity": pre_stats["conflict_intensity"],
            "tgd_pre_mean_cosine": pre_stats["mean_cosine"],
            "tgd_pre_min_cosine": pre_stats["min_cosine"],
            "tgd_pre_max_cosine": pre_stats["max_cosine"],
            "tgd_post_num_pairs": post_stats["num_pairs"],
            "tgd_post_neg_ratio": post_stats["neg_ratio"],
            "tgd_post_conflict_intensity": post_stats["conflict_intensity"],
            "tgd_post_mean_cosine": post_stats["mean_cosine"],
            "tgd_post_min_cosine": post_stats["min_cosine"],
            "tgd_post_max_cosine": post_stats["max_cosine"],
        }
        record["tgd_neg_ratio_reduction"] = (
            record["tgd_pre_neg_ratio"] - record["tgd_post_neg_ratio"]
        )
        record["tgd_conflict_intensity_reduction"] = (
            record["tgd_pre_conflict_intensity"]
            - record["tgd_post_conflict_intensity"]
        )

        self.loss_cache.update(
            {
                key: float(value)
                for key, value in record.items()
                if key != "step"
            }
        )

        if not self.is_world_process_zero():
            return
        os.makedirs(self._gradient_conflict_stats_dir(), exist_ok=True)
        metrics_path = os.path.join(
            self._gradient_conflict_stats_dir(), "metrics.jsonl"
        )
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _pcgrad(
        self,
        task_grads: list[list[torch.Tensor | None]],
        *,
        return_adjusted: bool = False,
    ) -> list[torch.Tensor | None] | tuple[list[torch.Tensor | None], list[list[torch.Tensor | None]]]:
        if not task_grads:
            return ([], []) if return_adjusted else []
        adjusted: list[list[torch.Tensor | None]] = []
        num_tasks = len(task_grads)
        for i in range(num_tasks):
            grad_i = [
                g.clone() if g is not None else None for g in task_grads[i]
            ]
            for j in range(num_tasks):
                if i == j:
                    continue
                dot = None
                norm = None
                for gi, gj in zip(grad_i, task_grads[j], strict=True):
                    if gi is None or gj is None:
                        continue
                    value = torch.sum(gi * gj)
                    dot = value if dot is None else dot + value
                    norm_value = torch.sum(gj * gj)
                    norm = norm_value if norm is None else norm + norm_value
                if dot is None or norm is None:
                    continue
                if dot.item() < 0.0 and norm.item() > 0.0:
                    coef = dot / norm
                    for idx, (gi, gj) in enumerate(zip(grad_i, task_grads[j], strict=True)):
                        if gi is None or gj is None:
                            continue
                        grad_i[idx] = gi - coef * gj
            adjusted.append(grad_i)

        merged: list[torch.Tensor | None] = []
        for param_idx in range(len(task_grads[0])):
            grads = [
                grad_list[param_idx]
                for grad_list in adjusted
                if grad_list[param_idx] is not None
            ]
            if not grads:
                merged.append(None)
                continue
            total = grads[0].clone()
            for g in grads[1:]:
                total.add_(g)
            total.div_(len(grads))
            merged.append(total)
        if return_adjusted:
            return merged, adjusted
        return merged

    def _apply_pcgrad(self) -> None:
        if not self._pcgrad_enabled():
            return
        shared_params = self._get_shared_params()
        if not shared_params:
            return
        task_grads = []
        for task_id, grads in self._task_grad_buffer.items():
            count = float(self._task_count_buffer.get(task_id, 0))
            if count <= 0:
                continue
            task_grads.append(
                [g / count if g is not None else None for g in grads]
            )
        if not task_grads:
            return
        with torch.no_grad():
            pre_stats = self._compute_gradient_conflict_stats(task_grads)
            if len(task_grads) == 1:
                merged = task_grads[0]
                adjusted = task_grads
            else:
                merged, adjusted = self._pcgrad(task_grads, return_adjusted=True)
            post_stats = self._compute_gradient_conflict_stats(adjusted)
            self._record_gradient_conflict_stats(
                pre_stats,
                post_stats,
                num_groups=len(task_grads),
            )
            for param, grad in zip(shared_params, merged, strict=True):
                if grad is None:
                    param.grad = None
                else:
                    if param.grad is None:
                        param.grad = grad.detach()
                    else:
                        param.grad.copy_(grad)

    def _apply_shared_expert_freeze(self) -> None:
        freeze_steps = self._shared_expert_freeze_steps()
        if freeze_steps <= 0:
            return
        shared_params = self._get_shared_expert_params()
        if not shared_params:
            return
        progress = min(self.state.global_step, freeze_steps) / float(freeze_steps)
        scale = max(0.0, 1.0 - progress)
        if scale >= 1.0:
            return
        with torch.no_grad():
            for param in shared_params:
                if param.grad is None:
                    continue
                param.grad.mul_(scale)

    def _feature_vis_enabled(self) -> bool:
        return bool(
            getattr(self.exp_config.trainer_config, "feature_vis_enable", False)
        )

    def _feature_vis_interval(self) -> int:
        return int(getattr(self.exp_config.trainer_config, "feature_vis_steps", 2000))

    def _feature_vis_target_groups(self) -> int:
        # Fixed 40-task stratified visualization.
        return 40

    def _feature_probe_batch_size(self) -> int:
        return int(
            getattr(self.exp_config.trainer_config, "feature_vis_probe_batch_size", 16)
        )

    def _feature_probe_use_images(self) -> bool:
        return bool(
            getattr(self.exp_config.trainer_config, "feature_vis_probe_use_images", False)
        )

    def _feature_vis_num_samples(self) -> int:
        return int(
            getattr(self.exp_config.trainer_config, "feature_vis_num_samples", 320)
        )

    def _feature_vis_max_scan(self) -> int:
        return int(
            getattr(self.exp_config.trainer_config, "feature_vis_max_scan", 4000)
        )

    def _feature_vis_dir(self) -> str:
        out_dir = getattr(
            self.exp_config.trainer_config, "feature_vis_output_dir", ""
        ).strip()
        if out_dir:
            return out_dir
        return os.path.join(self.args.output_dir, "feature_vis")

    @staticmethod
    def _hash_prompt_from_instance(instance: dict) -> Optional[str]:
        if not isinstance(instance, dict):
            return None
        input_ids = instance.get("input_ids")
        if input_ids is None:
            return None
        if not torch.is_tensor(input_ids):
            input_ids = torch.as_tensor(input_ids)
        ids = input_ids.detach().to("cpu").view(-1)
        attention_mask = instance.get("attention_mask")
        if attention_mask is not None:
            if not torch.is_tensor(attention_mask):
                attention_mask = torch.as_tensor(attention_mask)
            mask = attention_mask.detach().to("cpu").bool().view(-1)
            if mask.numel() == ids.numel():
                ids = ids[mask]
        if ids.numel() == 0:
            return "empty"
        digest = hashlib.blake2b(
            ids.to(torch.int64).contiguous().numpy().tobytes(), digest_size=8
        ).hexdigest()
        return digest

    def _build_feature_vis_reference(self) -> bool:
        if self.train_dataset is None:
            return False
        dataset_len = len(self.train_dataset)
        if dataset_len <= 0:
            return False
        target_groups = max(1, self._feature_vis_target_groups())
        num_samples = max(self._feature_vis_num_samples(), target_groups)
        max_scan = max(self._feature_vis_max_scan(), num_samples)
        num_scan = min(dataset_len, max_scan)
        target_groups = min(target_groups, num_scan)
        per_group = max(1, num_samples // target_groups)

        generator = torch.Generator()
        generator.manual_seed(2026)
        scan_indices = torch.randperm(dataset_len, generator=generator)[:num_scan].tolist()

        group_to_indices: dict[str, list[int]] = {}
        index_to_group: dict[int, str] = {}
        total_selected = 0
        for idx in scan_indices:
            try:
                item = self.train_dataset[idx]
            except Exception:
                continue
            group_key = self._hash_prompt_from_instance(item)
            if group_key is None:
                continue
            index_to_group[idx] = group_key
            if group_key not in group_to_indices and len(group_to_indices) >= target_groups:
                continue
            bucket = group_to_indices.setdefault(group_key, [])
            if len(bucket) >= per_group:
                continue
            bucket.append(idx)
            total_selected += 1
            if total_selected >= target_groups * per_group:
                break

        selected_indices: list[int] = []
        selected_keys: list[str] = []
        for group_key in sorted(group_to_indices.keys()):
            for idx in group_to_indices[group_key]:
                selected_indices.append(idx)
                selected_keys.append(group_key)

        selected_set = set(selected_indices)
        for idx in scan_indices:
            if len(selected_indices) >= num_samples:
                break
            if idx in selected_set:
                continue
            group_key = index_to_group.get(idx)
            if group_key is None or group_key not in group_to_indices:
                continue
            selected_indices.append(idx)
            selected_keys.append(group_key)
            selected_set.add(idx)

        if not selected_indices:
            return False

        unique_keys = sorted(set(selected_keys))
        self._feature_vis_label_map = {key: i for i, key in enumerate(unique_keys)}
        self._feature_vis_ref_indices = selected_indices
        self._feature_vis_ref_keys = selected_keys
        self._feature_vis_pca_mean = None
        self._feature_vis_pca_basis = None
        if self.is_world_process_zero():
            logger.info(
                "Feature vis reference initialized: samples={}, prompt_groups={} (target=40)",
                len(selected_indices),
                len(unique_keys),
            )
        return True

    def _collect_feature_batch(self) -> Optional[tuple[dict, list[str]]]:
        if self._feature_vis_ref_indices is None or self._feature_vis_ref_keys is None:
            if not self._build_feature_vis_reference():
                return None
        assert self._feature_vis_ref_indices is not None
        assert self._feature_vis_ref_keys is not None

        instances = []
        prompt_keys: list[str] = []
        for idx, group_key in zip(self._feature_vis_ref_indices, self._feature_vis_ref_keys):
            if idx >= len(self.train_dataset):
                continue
            try:
                item = self.train_dataset[idx]
            except Exception:
                continue
            instances.append(item)
            prompt_keys.append(group_key)
        if not instances:
            return None

        batch = self.data_collator(instances)
        batch = self._prepare_inputs(batch)
        required = ("input_ids", "attention_mask", "states")
        if self._feature_probe_use_images():
            required += ("images", "image_masks")
        if not all(key in batch for key in required):
            return None
        if len(prompt_keys) != int(batch["input_ids"].shape[0]):
            prompt_keys = self._build_prompt_hash_keys(
                batch["input_ids"], batch.get("attention_mask")
            )
        return batch, prompt_keys

    def _fit_feature_vis_pca_basis(self, features: torch.Tensor) -> None:
        x = torch.nan_to_num(features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-1e4, max=1e4)
        mean = x.mean(dim=0, keepdim=True)
        centered = x - mean
        dim = centered.shape[1]
        basis = torch.zeros((dim, 2), dtype=torch.float32)
        q = min(2, centered.shape[0] - 1, dim)
        if q > 0:
            try:
                _, _, v = torch.pca_lowrank(centered, q=q)
                basis[:, :q] = v[:, :q].float()
            except Exception:
                basis[0, 0] = 1.0
                if dim > 1:
                    basis[1, 1] = 1.0
        else:
            basis[0, 0] = 1.0
        self._feature_vis_pca_mean = mean.cpu()
        self._feature_vis_pca_basis = basis.cpu()

    def _pca_2d(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2:
            raise ValueError("features must be 2D")
        if features.shape[0] < 1:
            return torch.zeros((features.shape[0], 2), dtype=features.dtype)
        x = torch.nan_to_num(features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        x = torch.clamp(x, min=-1e4, max=1e4)
        if (
            self._feature_vis_pca_mean is None
            or self._feature_vis_pca_basis is None
            or self._feature_vis_pca_basis.shape[0] != x.shape[1]
        ):
            self._fit_feature_vis_pca_basis(x)
        assert self._feature_vis_pca_mean is not None
        assert self._feature_vis_pca_basis is not None
        mean = self._feature_vis_pca_mean.to(device=x.device, dtype=x.dtype)
        basis = self._feature_vis_pca_basis.to(device=x.device, dtype=x.dtype)
        proj = (x - mean) @ basis
        if proj.shape[1] < 2:
            zeros = torch.zeros((proj.shape[0], 2 - proj.shape[1]), dtype=proj.dtype, device=proj.device)
            proj = torch.cat([proj, zeros], dim=1)
        return proj[:, :2]

    @staticmethod
    def _compute_cluster_metrics(
        features: torch.Tensor, labels: torch.Tensor
    ) -> dict[str, float]:
        features = torch.nan_to_num(features.float(), nan=0.0, posinf=1e4, neginf=-1e4)
        features = torch.clamp(features, min=-1e4, max=1e4)
        unique_labels = torch.unique(labels)
        centers = []
        intra_vars = []
        for label in unique_labels:
            group_features = features[labels == label]
            if group_features.shape[0] == 0:
                continue
            center = group_features.mean(dim=0)
            centers.append(center)
            if group_features.shape[0] > 1:
                var = ((group_features - center) ** 2).sum(dim=1).mean()
                intra_vars.append(var)

        intra_var = float(torch.stack(intra_vars).mean().item()) if intra_vars else 0.0
        inter_dist = 0.0
        if len(centers) > 1:
            center_tensor = torch.stack(centers, dim=0).double()
            dist_mat = torch.cdist(center_tensor, center_tensor, p=2).float()
            dist_mat = torch.nan_to_num(
                dist_mat, nan=0.0, posinf=1e6, neginf=0.0
            )
            n = dist_mat.shape[0]
            off_diag = ~torch.eye(n, dtype=torch.bool, device=dist_mat.device)
            off_values = dist_mat[off_diag]
            if off_values.numel() > 0:
                inter_dist = float(off_values.mean().item())
        sep_ratio = inter_dist / (intra_var**0.5 + 1e-8)
        return {
            "num_points": float(features.shape[0]),
            "num_groups": float(len(centers)),
            "intra_var": float(intra_var),
            "inter_dist": float(inter_dist),
            "sep_ratio": float(sep_ratio),
        }

    def _save_feature_metrics(self, metrics: dict[str, float]) -> None:
        os.makedirs(self._feature_vis_dir(), exist_ok=True)
        metrics_path = os.path.join(self._feature_vis_dir(), "metrics.jsonl")
        record = {"step": int(self.state.global_step)}
        record.update({k: float(v) for k, v in metrics.items()})
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

    @staticmethod
    def _pool_hidden_by_mask(
        hidden: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if mask is None:
            return hidden.mean(dim=1)
        mask = mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (hidden * mask).sum(dim=1) / denom

    def _feature_probe_time(self) -> float:
        value = os.getenv("DEXBOTIC_FEATURE_PROBE_T", "0.5").strip()
        try:
            probe_t = float(value)
        except ValueError:
            probe_t = 0.5
        return max(0.0, min(1.0, probe_t))

    def _build_probe_noise(
        self, shape: tuple[int, int, int], dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(2026)
        noise = torch.randn(shape, generator=generator, dtype=torch.float32, device="cpu")
        return noise.to(device=device, dtype=dtype)

    def _extract_suffix_probe_feature(
        self, model_ref, batch: dict[str, torch.Tensor]
    ) -> Optional[torch.Tensor]:
        states = batch.get("states")
        if states is None:
            return None
        total_batch = int(states.shape[0])
        probe_bs = max(1, int(self._feature_probe_batch_size()))
        use_images = self._feature_probe_use_images()
        chunk_size = int(getattr(model_ref.model.config, "chunk_size", 0))
        action_dim = int(getattr(model_ref.config, "action_dim", 0))
        if chunk_size <= 0 or action_dim <= 0:
            return None

        feature_chunks: list[torch.Tensor] = []
        probe_t = self._feature_probe_time()
        for start in range(0, total_batch, probe_bs):
            end = min(total_batch, start + probe_bs)
            states_chunk = states[start:end]
            input_ids_chunk = batch["input_ids"][start:end]
            attn_mask_chunk = batch["attention_mask"][start:end]
            task_ids = batch.get("task_ids")
            task_ids_chunk = task_ids[start:end] if task_ids is not None else None
            actions = batch.get("actions")
            actions_chunk = actions[start:end] if actions is not None else None
            images_chunk = batch.get("images")
            image_masks_chunk = batch.get("image_masks")
            if not use_images:
                images_chunk = None
                image_masks_chunk = None
            else:
                images_chunk = images_chunk[start:end] if images_chunk is not None else None
                image_masks_chunk = (
                    image_masks_chunk[start:end] if image_masks_chunk is not None else None
                )

            prefix_tokens, prefix_mask, prefix_ar_mask, input_tokens = model_ref.embed_prefix(
                input_ids=input_ids_chunk,
                attention_mask=attn_mask_chunk,
                images=images_chunk,
                image_masks=image_masks_chunk,
                return_input_tokens=True,
            )
            task_token = model_ref._compute_task_token(
                input_tokens=input_tokens,
                attention_mask=attn_mask_chunk,
                task_ids=task_ids_chunk,
            )

            noise = self._build_probe_noise(
                (states_chunk.shape[0], chunk_size, action_dim),
                states_chunk.dtype,
                states_chunk.device,
            )
            time = torch.full(
                (states_chunk.shape[0],),
                probe_t,
                dtype=states_chunk.dtype,
                device=states_chunk.device,
            )
            if actions_chunk is not None:
                actions_chunk = actions_chunk.to(
                    device=states_chunk.device, dtype=states_chunk.dtype
                )
                if actions_chunk.shape == noise.shape:
                    x_t = probe_t * noise + (1.0 - probe_t) * actions_chunk
                else:
                    x_t = noise
            else:
                x_t = noise

            suffix_tokens, suffix_mask, suffix_ar_mask = model_ref.embed_suffix(
                states_chunk, x_t, time, task_token=task_token
            )
            input_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
            ar_mask = torch.cat([prefix_ar_mask, suffix_ar_mask], dim=0)
            attn_mask = make_attn_mask_4d(make_attn_mask(input_mask, ar_mask))
            positions = torch.cumsum(input_mask, dim=1) - 1
            position_embeddings = model_ref.model.llm.rotary_emb(prefix_tokens, positions)

            (_, suffix_out), _, _, _ = model_ref._inner_forward_mot(
                [model_ref.model.llm, model_ref.model.action_expert],
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                position_embeddings=position_embeddings,
                past_key_values=None,
                cache_position=positions,
                output_hidden_states=False,
                output_attentions=False,
            )
            if suffix_out is None:
                return None
            suffix_ctx = suffix_out[:, -chunk_size:, :]
            pooled = self._pool_hidden_by_mask(suffix_ctx, None)
            feature_chunks.append(pooled)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not feature_chunks:
            return None
        return torch.cat(feature_chunks, dim=0)

 
    def _save_feature_vis(
        self,
        features_2d: torch.Tensor,
        labels: torch.Tensor,
        metrics: dict[str, float],
    ) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as exc:
            logger.warning("Feature visualization skipped: matplotlib unavailable ({})", exc)
            return
        os.makedirs(self._feature_vis_dir(), exist_ok=True)
        x = features_2d[:, 0].cpu().numpy()
        y = features_2d[:, 1].cpu().numpy()
        labels_np = labels.cpu().numpy().astype(np.int64)
        fig, ax = plt.subplots(figsize=(8, 6), dpi=140)
        cmap = plt.cm.get_cmap("hsv", int(labels_np.max()) + 1 if labels_np.size > 0 else 1)
        scatter = ax.scatter(x, y, c=labels_np, cmap=cmap, s=10, alpha=0.8, linewidths=0)
        ax.set_title(
            "Prompt-Group Feature PCA | "
            f"step={self.state.global_step} | "
            f"samples={len(labels_np)} | "
            f"groups={int(metrics['num_groups'])} | "
            f"intra={metrics['intra_var']:.4f} | inter={metrics['inter_dist']:.4f}"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("prompt_group_id")
        fig.tight_layout()
        img_path = os.path.join(
            self._feature_vis_dir(), f"step_{self.state.global_step:07d}.png"
        )
        fig.savefig(img_path)
        plt.close(fig)
        npz_path = os.path.join(
            self._feature_vis_dir(), f"step_{self.state.global_step:07d}.npz"
        )
        np.savez_compressed(
            npz_path,
            pca_2d=features_2d.cpu().numpy(),
            prompt_group_id=labels_np,
            metrics=np.array(
                [
                    metrics["num_points"],
                    metrics["num_groups"],
                    metrics["intra_var"],
                    metrics["inter_dist"],
                    metrics["sep_ratio"],
                ],
                dtype=np.float32,
            ),
        )
        logger.info("Saved feature visualization to {}", img_path)

    def _maybe_visualize_features(self) -> None:
        if not self._feature_vis_enabled():
            return
        interval = self._feature_vis_interval()
        step = int(self.state.global_step)
        if interval <= 0 or step <= 0 or step % interval != 0:
            return
        if self._last_feature_vis_step == step:
            return
        self._last_feature_vis_step = step
        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        hooks_ok = hasattr(model_ref, "embed_prefix") and hasattr(
            model_ref, "_compute_task_token"
        )
        hooks_ok = self._dist_all_true(hooks_ok)
        if not hooks_ok:
            if self.is_world_process_zero():
                logger.warning("Feature visualization skipped: model does not expose pi0 hooks")
            return
        sample_pack = self._collect_feature_batch()
        has_batch = self._dist_all_true(sample_pack is not None)
        if not has_batch:
            if self.is_world_process_zero():
                logger.warning("Feature visualization skipped: no valid stratified samples")
            return
        if sample_pack is None:
            return
        batch, prompt_keys = sample_pack
        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                contextual_feature = self._extract_suffix_probe_feature(
                    model_ref, batch
                )
                if contextual_feature is None:
                    logger.warning(
                        "Feature visualization skipped: suffix probe feature is None"
                    )
                    return
                features = contextual_feature.detach().float().cpu()
                if len(prompt_keys) != int(features.shape[0]):
                    prompt_keys = self._build_prompt_hash_keys(
                        batch["input_ids"], batch.get("attention_mask")
                    )
                labels = torch.tensor(
                    [self._feature_vis_label_map.get(key, -1) for key in prompt_keys],
                    dtype=torch.long,
                )
                finite_rows = torch.isfinite(features).all(dim=1)
                valid_rows = finite_rows & (labels >= 0)
                if not valid_rows.all().item():
                    dropped = int((~finite_rows).sum().item())
                    invalid_label = int((labels < 0).sum().item())
                    kept = int(valid_rows.sum().item())
                    if self.is_world_process_zero():
                        logger.warning(
                            "Feature visualization dropped non-finite={} invalid-label={} at step {} (kept={})",
                            dropped,
                            invalid_label,
                            step,
                            kept,
                        )
                    features = features[valid_rows]
                    labels = labels[valid_rows]
                if features.shape[0] < 2:
                    if self.is_world_process_zero():
                        logger.warning(
                            "Feature visualization skipped at step {}: valid samples < 2",
                            step,
                        )
                    return
                features_2d = self._pca_2d(features)
                metrics = self._compute_cluster_metrics(features, labels)
                if self.is_world_process_zero():
                    self._save_feature_vis(features_2d, labels, metrics)
                    self._save_feature_metrics(metrics)
                    logger.info(
                        "Feature clustering @ step {}: points={} groups={} intra_var={:.6f} inter_dist={:.6f} sep_ratio={:.6f}",
                        step,
                        int(metrics["num_points"]),
                        int(metrics["num_groups"]),
                        metrics["intra_var"],
                        metrics["inter_dist"],
                        metrics["sep_ratio"],
                    )
        except Exception as exc:
            if self.is_world_process_zero():
                logger.warning("Feature visualization failed at step {}: {}", step, exc)
        finally:
            if was_training:
                self.model.train()

    def _save_checkpoint(self, model, trial, metrics=None) -> None:
        logger.info(f"Saving checkpoint at step {self.state.global_step}")
        if getattr(self.added_args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(
                self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(
                    weight_to_save, os.path.join(
                        output_dir, 'mm_projector.bin'))

        else:
            super(DexboticTrainer, self)._save_checkpoint(model, trial)
            # Copy norm_stats.json to checkpoint directory after saving
            if self.args.local_rank == 0 or self.args.local_rank == -1:
                from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
                checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
                run_dir = self._get_output_dir(trial=trial)
                output_dir = os.path.join(run_dir, checkpoint_folder)
                self._copy_norm_stats_to_checkpoint(output_dir)

    def _copy_norm_stats_to_checkpoint(self, checkpoint_dir: str) -> None:
        """Copy norm_stats.json from main output directory to checkpoint directory"""
        
        main_output_dir = self.args.output_dir
        norm_stats_src = os.path.join(main_output_dir, "norm_stats.json")
        norm_stats_dst = os.path.join(checkpoint_dir, "norm_stats.json")
        
        if os.path.exists(norm_stats_src):
            try:
                shutil.copy2(norm_stats_src, norm_stats_dst)
                logger.info(f"Copied norm_stats.json to checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to copy norm_stats.json to checkpoint: {e}")

    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        if getattr(self.added_args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(DexboticTrainer, self)._save(output_dir, state_dict)

    def _link_exp_config(self) -> TrainingArguments:
        """Link the exp config to the trainer args"""
        linked_args = {
            "output_dir": self.exp_config.trainer_config.output_dir,
            "num_train_epochs": self.exp_config.trainer_config.num_train_epochs,
            "max_steps": self.exp_config.trainer_config.num_train_steps,
            "per_device_train_batch_size": self.exp_config.trainer_config.per_device_train_batch_size,
            "gradient_accumulation_steps": self.exp_config.trainer_config.gradient_accumulation_steps,
            "save_strategy": self.exp_config.trainer_config.save_strategy,
            "save_steps": self.exp_config.trainer_config.save_steps,
            "save_total_limit": self.exp_config.trainer_config.save_total_limit,
            "save_only_model": self.exp_config.trainer_config.save_only_model,
            "logging_steps": self.exp_config.trainer_config.logging_steps,
            "gradient_checkpointing": self.exp_config.trainer_config.gradient_checkpointing,
            "dataloader_num_workers": self.exp_config.trainer_config.dataloader_num_workers,
            # "model_max_length": self.exp_config.trainer_config.model_max_length,
            "bf16": self.exp_config.trainer_config.bf16,
            "tf32": self.exp_config.trainer_config.tf32,
            "lr_scheduler_type": self.exp_config.trainer_config.lr_scheduler_type,
            "lr_scheduler_kwargs": self.exp_config.trainer_config.lr_scheduler_kwargs,
            "run_name": self.exp_config.trainer_config.run_name,
            'remove_unused_columns': False,
            "deepspeed": self.exp_config.trainer_config.deepspeed,
            "optim": self.exp_config.optimizer_config.optim,
            "learning_rate": self.exp_config.optimizer_config.base_lr,
            "adam_beta1": self.exp_config.optimizer_config.adam_beta1,
            "adam_beta2": self.exp_config.optimizer_config.adam_beta2,
            "adam_epsilon": self.exp_config.optimizer_config.adam_epsilon,
            "warmup_steps": self.exp_config.optimizer_config.warmup_steps,
            "warmup_ratio": self.exp_config.optimizer_config.warmup_ratio,
            "weight_decay": self.exp_config.optimizer_config.weight_decay,
            "max_grad_norm": self.exp_config.optimizer_config.clip_gradient_norm,
        }
        self.added_args = EasyDict({
            "tune_mm_mlp_adapter": self.exp_config.trainer_config.tune_mm_mlp_adapter,
        })
        training_args = TrainingArguments(**linked_args)
        return training_args

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not (self._task_aware_enabled() and self._pcgrad_enabled()):
            return super().training_step(model, inputs)

        model.train()
        inputs = self._prepare_inputs(inputs)
        group_keys = self._build_group_keys(inputs)
        if group_keys is None:
            return super().training_step(model, inputs)

        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        per_sample_loss = None
        if isinstance(outputs, dict):
            per_sample_loss = outputs.get("flow_matching_loss_per_sample")
        else:
            per_sample_loss = getattr(outputs, "flow_matching_loss_per_sample", None)
        if per_sample_loss is None:
            return super().training_step(model, inputs)

        shared_params = self._get_shared_params()
        if not shared_params:
            return super().training_step(model, inputs)

        grad_acc_steps = max(1, int(self.args.gradient_accumulation_steps))
        per_sample_loss = per_sample_loss.view(-1)
        if len(group_keys) != int(per_sample_loss.shape[0]):
            return super().training_step(model, inputs)
        scale = 1.0 / float(grad_acc_steps)

        group_to_indices: dict[str, list[int]] = {}
        for sample_idx, key in enumerate(group_keys):
            group_to_indices.setdefault(str(key), []).append(sample_idx)

        for group_key, sample_indices in group_to_indices.items():
            count = len(sample_indices)
            if count == 0:
                continue
            index_tensor = torch.tensor(
                sample_indices, device=per_sample_loss.device, dtype=torch.long
            )
            loss_task = per_sample_loss.index_select(0, index_tensor).mean() * scale
            grads = torch.autograd.grad(
                loss_task,
                shared_params,
                retain_graph=True,
                allow_unused=True,
            )
            grads = [g * count if g is not None else None for g in grads]
            if group_key not in self._task_grad_buffer:
                self._task_grad_buffer[group_key] = [
                    g.detach().clone() if g is not None else None for g in grads
                ]
                self._task_count_buffer[group_key] = count
            else:
                stored = self._task_grad_buffer[group_key]
                for idx, g in enumerate(grads):
                    if g is None:
                        continue
                    if stored[idx] is None:
                        stored[idx] = g.detach().clone()
                    else:
                        stored[idx].add_(g)
                self._task_count_buffer[group_key] += count

        if grad_acc_steps > 1:
            loss = loss / float(grad_acc_steps)
        self.accelerator.backward(loss)
        return loss.detach()

    def optimizer_step(self, *args, **kwargs):
        if self._pcgrad_enabled() and self.accelerator.sync_gradients:
            self._apply_pcgrad()
        if self.accelerator.sync_gradients:
            self._apply_shared_expert_freeze()
        result = super().optimizer_step(*args, **kwargs)
        if self.accelerator.sync_gradients:
            self._task_grad_buffer.clear()
            self._task_count_buffer.clear()
        return result

    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        loss_keys = [_ for _ in outputs if _.endswith("_loss")]
        norm_keys = [
            key
            for key in ("moe_norm", "router_norm", "total_param_norm")
            if key in outputs
        ]

        for loss_key in loss_keys + norm_keys:
            value = outputs[loss_key]
            if value is None:
                if loss_key not in self.loss_cache:
                    self.loss_cache[loss_key] = 0.0
                continue
            if torch.is_tensor(value):
                if torch.isclose(value, torch.zeros_like(value)):
                    if loss_key not in self.loss_cache:
                        self.loss_cache[loss_key] = 0.0
                    continue
                self.loss_cache[loss_key] = value.detach().item()
            else:
                self.loss_cache[loss_key] = float(value)
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        logs.update(self.loss_cache)
        super().log(logs, start_time)
        self._maybe_visualize_features()


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str) -> None:
    """Collects the state dict and dump to disk."""

    if getattr(trainer.added_args, "tune_mm_mlp_adapter", False):
        keys_to_match = ['mm_projector']
        weight_to_save_mm_projector = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match)

        trainer.model.config.save_pretrained(output_dir)
        trainer.processing_class.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(
                        mm_projector_folder,
                        f'{current_folder}.bin'))

            else:
                torch.save(
                    weight_to_save_mm_projector,
                    os.path.join(
                        output_dir,
                        'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa
