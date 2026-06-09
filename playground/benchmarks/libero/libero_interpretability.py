#!/usr/bin/env python3
import argparse
import importlib.util
import json
import math
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    import megfile  # type: ignore
except ImportError:
    class _MegfileFallback:
        @staticmethod
        def smart_open(path: str, mode: str = "r"):
            return open(path, mode, encoding=None if "b" in mode else "utf-8")

        @staticmethod
        def smart_exists(path: str) -> bool:
            return os.path.exists(path)

    megfile = _MegfileFallback()

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexbotic.data.collator import DataCollatorForSupervisedDataset


def parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def parse_optional_bool(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    return parse_bool(value)


def load_module_from_path(script_path: str):
    module_path = Path(script_path).resolve()
    spec = importlib.util.spec_from_file_location("libero_pi0_module", str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load script: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_prompt_from_item(item: Any) -> Optional[str]:
    if not isinstance(item, dict):
        return None
    for key in (
        "prompt",
        "task_prompt",
        "instruction",
        "task_description",
        "language",
        "task",
        "task_name",
    ):
        if key in item:
            return normalize_prompt_value(item[key])
    if "conversations" in item:
        return normalize_prompt_value(item["conversations"])
    return None


def normalize_prompt_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, list):
        if not value:
            return None
        return normalize_prompt_value(value[0])
    if isinstance(value, dict):
        if "value" in value:
            return normalize_prompt_value(value["value"])
        return None
    prompt = str(value).strip()
    return prompt or None


def read_prompt_from_jsonl(jsonl_file: str) -> Optional[str]:
    try:
        with megfile.smart_open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt = extract_prompt_from_item(item)
                if prompt is not None:
                    return prompt
    except Exception:
        return None
    return None


def _build_task_name_map_from_dataset(dataset) -> dict[int, str]:
    prompt_to_id = getattr(dataset, "task_prompt_to_id", None)
    if isinstance(prompt_to_id, dict) and prompt_to_id:
        id_to_prompt = {int(v): str(k) for k, v in prompt_to_id.items()}
        return id_to_prompt
    return {}


def _dataset_has_task_id(dataset) -> bool:
    try:
        item = dataset[0]
    except Exception:
        return False
    return isinstance(item, dict) and "task_id" in item


def _derive_task_ids_from_dataset(dataset):
    if not hasattr(dataset, "global_index") or not hasattr(dataset, "file_name_map"):
        raise RuntimeError("Dataset does not expose global_index/file_name_map for task_id derivation.")

    file_prompt_map: dict[str, str] = {}
    for jsonl_file in dataset.file_name_map.values():
        prompt = read_prompt_from_jsonl(jsonl_file)
        if prompt is None:
            prompt = "__unknown__"
        file_prompt_map[jsonl_file] = prompt

    prompt_list = sorted(set(file_prompt_map.values()))
    prompt_to_id = {prompt: idx for idx, prompt in enumerate(prompt_list)}
    id_to_prompt = {idx: prompt for prompt, idx in prompt_to_id.items()}

    sample_task_ids = []
    for _dataset_idx, file_idx, _frame_idx in dataset.global_index:
        jsonl_file = dataset.file_name_map[file_idx]
        prompt = file_prompt_map.get(jsonl_file, "__unknown__")
        sample_task_ids.append(int(prompt_to_id[prompt]))

    return sample_task_ids, id_to_prompt


class TaskIdWrappedDataset(Dataset):
    def __init__(self, base_dataset: Dataset, sample_task_ids: list[int]):
        self.base_dataset = base_dataset
        self.sample_task_ids = sample_task_ids

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.base_dataset[idx]
        if not isinstance(item, dict):
            raise RuntimeError("Dataset item must be a dict.")
        output = dict(item)
        output["task_id"] = torch.tensor(int(self.sample_task_ids[idx]), dtype=torch.long)
        return output


def ensure_task_ids(dataset):
    id_to_name = _build_task_name_map_from_dataset(dataset)
    if _dataset_has_task_id(dataset):
        return dataset, id_to_name
    sample_task_ids, derived_id_to_name = _derive_task_ids_from_dataset(dataset)
    if not id_to_name:
        id_to_name = derived_id_to_name
    return TaskIdWrappedDataset(dataset, sample_task_ids), id_to_name


class InterpretabilityCollator:
    def __init__(self, base_collator: DataCollatorForSupervisedDataset):
        self.base_collator = base_collator

    def __call__(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        batch = self.base_collator(instances)
        if "task_id" in instances[0] and "task_ids" not in batch:
            task_ids = [x["task_id"] for x in instances]
            if torch.is_tensor(task_ids[0]):
                batch["task_ids"] = torch.stack(task_ids)
            else:
                batch["task_ids"] = torch.tensor(task_ids, dtype=torch.long)
        return batch


class InterpretabilityCollector:
    def __init__(
        self,
        model,
        *,
        num_pseudo_experts: int,
        collect_enabled: bool,
        save_moe_raw_limit_batches: int = 0,
        save_token_routes_limit_batches: int = 32,
    ):
        self.model = model
        self.collect_enabled = bool(collect_enabled)
        self.num_pseudo_experts = max(int(num_pseudo_experts), 1)
        self.save_moe_raw_limit_batches = max(int(save_moe_raw_limit_batches), 0)
        self.save_token_routes_limit_batches = max(int(save_token_routes_limit_batches), 0)

        self.mode: Optional[str] = None
        self.hooks = []

        self.current_task_ids: Optional[np.ndarray] = None
        self.current_layer_indices: list[int] = []
        self.current_router_by_layer: list[np.ndarray] = []
        self.current_top1_by_layer: list[np.ndarray] = []
        self.current_topk_by_layer: list[np.ndarray] = []
        self.current_pseudo_by_layer: list[np.ndarray] = []

        self.task_router_sum: dict[int, np.ndarray] = {}
        self.task_top1_sum: dict[int, np.ndarray] = {}
        self.task_topk_sum: dict[int, np.ndarray] = {}
        self.task_pseudo_sum: dict[int, np.ndarray] = {}
        self.task_counts: dict[int, int] = defaultdict(int)
        self.task_layer_top1_sum: dict[int, dict[int, np.ndarray]] = {}
        self.task_layer_counts: dict[int, dict[int, int]] = {}

        self.raw_records: list[dict[str, Any]] = []
        self.token_route_records: list[dict[str, Any]] = []
        self.batch_index = 0
        self.expert_dim: Optional[int] = None

        self._register_hooks()

    def _register_hooks(self) -> None:
        action_expert = getattr(getattr(self.model, "model", None), "action_expert", None)
        layers = getattr(action_expert, "layers", None)
        if layers is None:
            raise RuntimeError("Model does not expose model.action_expert.layers")

        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            def _hook(module, inputs, _layer_idx=layer_idx):
                self._on_mlp_forward_pre(_layer_idx, module, inputs)

            handle = mlp.register_forward_pre_hook(_hook)
            self.hooks.append(handle)

    def close(self) -> None:
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def start_batch(self, task_ids: torch.Tensor) -> None:
        self.current_task_ids = task_ids.detach().view(-1).cpu().numpy().astype(np.int64)
        self.current_layer_indices = []
        self.current_router_by_layer = []
        self.current_top1_by_layer = []
        self.current_topk_by_layer = []
        self.current_pseudo_by_layer = []

    def _set_mode(self, mode: str) -> None:
        if self.mode is None:
            self.mode = mode

    def _on_mlp_forward_pre(self, layer_idx: int, module, inputs) -> None:
        if not self.collect_enabled:
            return
        if self.current_task_ids is None:
            return
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x):
            return

        with torch.no_grad():
            if self._is_moe_module(module):
                self._set_mode("moe")
                router_usage, top1_usage, topk_usage, raw_router, raw_top1, raw_topk = (
                    self._collect_moe_usage(module, x)
                )
                self.current_layer_indices.append(layer_idx)
                self.current_router_by_layer.append(router_usage)
                self.current_top1_by_layer.append(top1_usage)
                self.current_topk_by_layer.append(topk_usage)
                if self.batch_index < self.save_token_routes_limit_batches:
                    self.token_route_records.append(
                        {
                            "batch": int(self.batch_index),
                            "layer": int(layer_idx),
                            "task_ids": self.current_task_ids.copy(),
                            "top1_idx": raw_top1,
                        }
                    )
                if self.batch_index < self.save_moe_raw_limit_batches:
                    self.raw_records.append(
                        {
                            "batch": int(self.batch_index),
                            "layer": int(layer_idx),
                            "task_ids": self.current_task_ids.copy(),
                            "router_probs": raw_router,
                            "top1_idx": raw_top1,
                            "topk_idx": raw_topk,
                        }
                    )
            else:
                self._set_mode("pseudo")
                pseudo_usage = self._collect_pseudo_usage(module, x)
                self.current_pseudo_by_layer.append(pseudo_usage)

    @staticmethod
    def _is_moe_module(module) -> bool:
        return bool(getattr(module, "use_moe", False)) and getattr(module, "moe_ffn", None) is not None

    def _collect_moe_usage(self, module, x: torch.Tensor):
        moe_ffn = module.moe_ffn
        batch_size, seq_len, hidden_dim = x.shape
        if hasattr(moe_ffn, "_routing"):
            _x_mat, routing, index, _expert_choice, _moe_info, shape = moe_ffn._routing(x)
            batch_size, seq_len = shape
            num_experts = int(getattr(moe_ffn, "num_experts", routing.shape[-1]))
            router_probs = routing.view(batch_size, seq_len, num_experts)
            top1_idx = index.view(batch_size, seq_len)
            topk_idx = top1_idx.unsqueeze(-1)
        else:
            w_gating = getattr(moe_ffn, "w_gating", None)
            if w_gating is None:
                raise RuntimeError(
                    "MoE FFN does not expose w_gating or _routing for route collection"
                )
            num_experts = int(getattr(moe_ffn, "num_experts", w_gating.shape[-1]))
            top_k = int(getattr(moe_ffn, "top_k", getattr(module, "moe_top_k", 1)))
            top_k = max(1, min(top_k, num_experts))
            x_flat = x.reshape(-1, hidden_dim)
            logits = torch.matmul(x_flat.float(), w_gating.float())
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            router_probs = torch.softmax(logits, dim=-1).view(batch_size, seq_len, num_experts)
            router_probs = torch.nan_to_num(
                router_probs, nan=0.0, posinf=0.0, neginf=0.0
            )
            topk_idx = torch.topk(router_probs, k=top_k, dim=-1).indices
            top1_idx = topk_idx[..., 0]
        self.expert_dim = num_experts

        top1_one_hot = F.one_hot(top1_idx, num_experts).float()
        top1_usage = top1_one_hot.mean(dim=1)

        one_hot = F.one_hot(topk_idx, num_experts).float()
        topk_counts = one_hot.sum(dim=(1, 2))
        topk_usage = topk_counts / topk_counts.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        router_usage = router_probs.mean(dim=1)
        return (
            router_usage.cpu().numpy(),
            top1_usage.cpu().numpy(),
            topk_usage.cpu().numpy(),
            router_probs.cpu().numpy(),
            top1_idx.cpu().numpy(),
            topk_idx.cpu().numpy(),
        )

    def _collect_pseudo_usage(self, module, x: torch.Tensor) -> np.ndarray:
        gate_proj = getattr(module, "gate_proj", None)
        up_proj = getattr(module, "up_proj", None)
        act_fn = getattr(module, "act_fn", None)
        if gate_proj is None or up_proj is None or act_fn is None:
            raise RuntimeError("Pseudo-expert mode requires gate_proj/up_proj/act_fn on MLP module")

        hidden = act_fn(gate_proj(x.float())) * up_proj(x.float())
        batch_size, seq_len, hidden_dim = hidden.shape

        expert_dim = min(self.num_pseudo_experts, hidden_dim)
        self.expert_dim = expert_dim

        boundaries = np.linspace(0, hidden_dim, num=expert_dim + 1, dtype=np.int64)
        energies = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if end <= start:
                continue
            energies.append(hidden[..., start:end].pow(2).sum(dim=-1))
        if not energies:
            raise RuntimeError("Failed to build pseudo-expert groups from FFN hidden dimension")

        energy = torch.stack(energies, dim=-1)
        probs = energy / energy.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        usage = probs.mean(dim=1)
        return usage.cpu().numpy()

    def end_batch(self) -> None:
        if not self.collect_enabled:
            self.batch_index += 1
            return
        if self.current_task_ids is None:
            self.batch_index += 1
            return

        if self.mode == "moe":
            if not self.current_router_by_layer:
                self.batch_index += 1
                return
            router_usage = np.mean(np.stack(self.current_router_by_layer, axis=0), axis=0)
            top1_usage = np.mean(np.stack(self.current_top1_by_layer, axis=0), axis=0)
            topk_usage = np.mean(np.stack(self.current_topk_by_layer, axis=0), axis=0)
            for idx, task_id in enumerate(self.current_task_ids.tolist()):
                task_id = int(task_id)
                if task_id not in self.task_router_sum:
                    self.task_router_sum[task_id] = np.zeros(router_usage.shape[-1], dtype=np.float64)
                if task_id not in self.task_top1_sum:
                    self.task_top1_sum[task_id] = np.zeros(top1_usage.shape[-1], dtype=np.float64)
                if task_id not in self.task_topk_sum:
                    self.task_topk_sum[task_id] = np.zeros(topk_usage.shape[-1], dtype=np.float64)
                self.task_router_sum[task_id] += router_usage[idx].astype(np.float64)
                self.task_top1_sum[task_id] += top1_usage[idx].astype(np.float64)
                self.task_topk_sum[task_id] += topk_usage[idx].astype(np.float64)
                self.task_counts[task_id] += 1
            for layer_idx, layer_top1_usage in zip(
                self.current_layer_indices, self.current_top1_by_layer
            ):
                layer_sum = self.task_layer_top1_sum.setdefault(int(layer_idx), {})
                layer_counts = self.task_layer_counts.setdefault(int(layer_idx), {})
                for idx, task_id in enumerate(self.current_task_ids.tolist()):
                    task_id = int(task_id)
                    if task_id not in layer_sum:
                        layer_sum[task_id] = np.zeros(
                            layer_top1_usage.shape[-1], dtype=np.float64
                        )
                    layer_sum[task_id] += layer_top1_usage[idx].astype(np.float64)
                    layer_counts[task_id] = layer_counts.get(task_id, 0) + 1
        else:
            if not self.current_pseudo_by_layer:
                self.batch_index += 1
                return
            pseudo_usage = np.mean(np.stack(self.current_pseudo_by_layer, axis=0), axis=0)
            for idx, task_id in enumerate(self.current_task_ids.tolist()):
                task_id = int(task_id)
                if task_id not in self.task_pseudo_sum:
                    self.task_pseudo_sum[task_id] = np.zeros(pseudo_usage.shape[-1], dtype=np.float64)
                self.task_pseudo_sum[task_id] += pseudo_usage[idx].astype(np.float64)
                self.task_counts[task_id] += 1

        self.batch_index += 1

    def _build_matrix_from_sum(
        self,
        task_sum: dict[int, np.ndarray],
        task_counts: Optional[dict[int, int]] = None,
    ):
        if task_counts is None:
            task_counts = self.task_counts
        task_ids = sorted(task_sum.keys())
        if not task_ids:
            raise RuntimeError("No interpretability samples were collected.")

        raw_mass = []
        counts = []
        for task_id in task_ids:
            count = int(task_counts.get(task_id, 0))
            if count <= 0:
                continue
            raw_mass.append(task_sum[task_id] / float(count))
            counts.append(count)

        matrix = np.asarray(raw_mass, dtype=np.float64)
        row_sum = matrix.sum(axis=1, keepdims=True)
        matrix = matrix / np.clip(row_sum, a_min=1e-12, a_max=None)

        return task_ids, np.asarray(counts, dtype=np.int64), matrix

    def finalize(self):
        mode = self.mode or "unknown"
        if mode == "moe":
            task_ids, counts, matrix = self._build_matrix_from_sum(self.task_top1_sum)
            router_prob_matrix = None
            if self.task_router_sum:
                _, _, router_prob_matrix = self._build_matrix_from_sum(self.task_router_sum)
            topk_matrix = None
            if self.task_topk_sum:
                _, _, topk_matrix = self._build_matrix_from_sum(self.task_topk_sum)
            layer_top1_matrices = []
            for layer_idx in sorted(self.task_layer_top1_sum.keys()):
                layer_task_ids, layer_counts, layer_matrix = self._build_matrix_from_sum(
                    self.task_layer_top1_sum[layer_idx],
                    self.task_layer_counts.get(layer_idx, {}),
                )
                layer_top1_matrices.append(
                    {
                        "layer": int(layer_idx),
                        "task_ids": layer_task_ids,
                        "counts": layer_counts,
                        "matrix": layer_matrix,
                    }
                )
        elif mode == "pseudo":
            task_ids, counts, matrix = self._build_matrix_from_sum(self.task_pseudo_sum)
            router_prob_matrix = None
            topk_matrix = None
            layer_top1_matrices = []
        else:
            raise RuntimeError("Collector did not receive any MLP hooks.")

        return {
            "mode": mode,
            "task_ids": task_ids,
            "counts": counts,
            "matrix": matrix,
            "router_prob_matrix": router_prob_matrix,
            "topk_matrix": topk_matrix,
            "layer_top1_matrices": layer_top1_matrices,
            "raw_records": self.raw_records,
            "token_route_records": self.token_route_records,
        }


def compute_pairwise_js(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    js = np.zeros((n, n), dtype=np.float64)
    eps = 1e-12
    for i in range(n):
        p = matrix[i]
        for j in range(i + 1, n):
            q = matrix[j]
            m = 0.5 * (p + q)
            kl_pm = np.sum(p * (np.log(p + eps) - np.log(m + eps)))
            kl_qm = np.sum(q * (np.log(q + eps) - np.log(m + eps)))
            val = 0.5 * (kl_pm + kl_qm)
            js[i, j] = val
            js[j, i] = val
    return js


def compute_metrics(matrix: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    eps = 1e-12
    entropy_per_task = -np.sum(matrix * np.log(matrix + eps), axis=1)

    pairwise_js = compute_pairwise_js(matrix)
    if matrix.shape[0] > 1:
        off_diag = pairwise_js[~np.eye(matrix.shape[0], dtype=bool)]
        pairwise_js_mean = float(off_diag.mean())
    else:
        pairwise_js_mean = 0.0

    row_mass = matrix * counts[:, None]
    joint = row_mass / np.clip(row_mass.sum(), a_min=eps, a_max=None)
    p_t = joint.sum(axis=1, keepdims=True)
    p_e = joint.sum(axis=0, keepdims=True)
    mi = np.sum(joint * (np.log(joint + eps) - np.log(p_t + eps) - np.log(p_e + eps)))

    weighted_entropy = float(np.sum((counts / np.clip(counts.sum(), eps, None)) * entropy_per_task))
    global_usage = row_mass.sum(axis=0)
    global_usage = global_usage / np.clip(global_usage.sum(), a_min=eps, a_max=None)
    global_entropy = -float(np.sum(global_usage * np.log(global_usage + eps)))
    normalized_global_entropy = (
        global_entropy / float(np.log(matrix.shape[1])) if matrix.shape[1] > 1 else 1.0
    )
    effective_experts = float(np.exp(global_entropy))

    return {
        "num_tasks": int(matrix.shape[0]),
        "num_experts": int(matrix.shape[1]),
        "mean_entropy": float(entropy_per_task.mean()),
        "weighted_entropy": weighted_entropy,
        "global_usage": global_usage.tolist(),
        "global_entropy": global_entropy,
        "normalized_global_entropy": float(normalized_global_entropy),
        "effective_experts": effective_experts,
        "max_expert_usage": float(global_usage.max(initial=0.0)),
        "min_expert_usage": float(global_usage.min()),
        "pairwise_js_mean": float(pairwise_js_mean),
        "mi_task_expert": float(mi),
        "entropy_per_task": entropy_per_task.tolist(),
        "pairwise_js_matrix": pairwise_js.tolist(),
    }


def cluster_task_order(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0], dtype=np.int64)

    try:
        from scipy.cluster.hierarchy import linkage, leaves_list
        from scipy.spatial.distance import pdist

        dist = pdist(matrix, metric="jensenshannon")
        dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
        if float(dist.max(initial=0.0)) <= 0.0:
            return np.arange(matrix.shape[0], dtype=np.int64)
        z = linkage(dist, method="average")
        order = leaves_list(z)
        return order.astype(np.int64)
    except Exception:
        entropy = -np.sum(matrix * np.log(np.clip(matrix, 1e-12, None)), axis=1)
        primary = np.argmax(matrix, axis=1)
        order = np.lexsort((entropy, primary))
        return order.astype(np.int64)


def _format_task_labels(task_ids: list[int], id_to_name: dict[int, str]) -> list[str]:
    labels = []
    for task_id in task_ids:
        task_name = id_to_name.get(int(task_id), f"task_{task_id}")
        labels.append(task_name)
    return labels


def save_heatmap(
    matrix: np.ndarray,
    task_labels: list[str],
    out_file: str,
    *,
    title: str,
    expert_prefix: str = "E",
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_tasks, num_experts = matrix.shape
    fig_w = max(7.0, 1.2 * num_experts)
    fig_h = max(5.0, 0.28 * num_tasks)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)

    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=float(matrix.max(initial=1e-8)))
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Task")

    ax.set_xticks(np.arange(num_experts))
    ax.set_xticklabels([f"{expert_prefix}{i}" for i in range(num_experts)], rotation=0)

    if num_tasks <= 60:
        ax.set_yticks(np.arange(num_tasks))
        ax.set_yticklabels(task_labels, fontsize=7)
    else:
        step = max(1, num_tasks // 40)
        ticks = np.arange(0, num_tasks, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([task_labels[i] for i in ticks], fontsize=7)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(expert | task)")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def save_task_barplots(
    matrix: np.ndarray,
    task_labels: list[str],
    out_dir: str,
    *,
    prefix: str,
    tasks_per_page: int = 16,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_tasks, num_experts = matrix.shape
    tasks_per_page = max(1, int(tasks_per_page))
    output_files = []

    page = 0
    for start in range(0, num_tasks, tasks_per_page):
        end = min(num_tasks, start + tasks_per_page)
        sub_labels = task_labels[start:end]
        sub_matrix = matrix[start:end]

        n = end - start
        cols = 4
        rows = int(math.ceil(n / float(cols)))
        fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 2.6 * rows), dpi=140)
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = np.array([axes])
        elif cols == 1:
            axes = np.array([[ax] for ax in axes])

        for idx in range(rows * cols):
            r = idx // cols
            c = idx % cols
            ax = axes[r, c]
            if idx >= n:
                ax.axis("off")
                continue
            values = sub_matrix[idx]
            ax.bar(np.arange(num_experts), values)
            ax.set_ylim(0.0, max(1.0, float(values.max(initial=0.0) * 1.1)))
            ax.set_title(sub_labels[idx], fontsize=8)
            ax.set_xlabel("Expert", fontsize=7)
            ax.set_ylabel("Prob", fontsize=7)
            ax.tick_params(axis="both", labelsize=7)

        fig.tight_layout()
        page += 1
        output_path = os.path.join(out_dir, f"{prefix}_task_bars_{page:03d}.png")
        fig.savefig(output_path)
        plt.close(fig)
        output_files.append(output_path)

    return output_files


def save_pairwise_js_heatmap(js_matrix: np.ndarray, task_labels: list[str], out_file: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = js_matrix.shape[0]
    fig_w = max(6.0, 0.24 * n)
    fig_h = max(5.0, 0.24 * n)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    image = ax.imshow(js_matrix, aspect="auto", cmap="viridis")
    ax.set_title("Pairwise Jensen-Shannon Divergence")
    ax.set_xlabel("Task")
    ax.set_ylabel("Task")

    if n <= 60:
        ticks = np.arange(n)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(task_labels, rotation=90, fontsize=6)
        ax.set_yticklabels(task_labels, fontsize=6)

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("JS")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        else:
            out[key] = value
    return out


def apply_model_variant(exp, variant: Optional[str]) -> None:
    if not variant:
        return
    variant = variant.strip().lower().replace("-", "_")
    presets = {
        "vanilla_moe": {
            "use_moe": True,
            "moe_type": "vanilla",
            "task_token_enable": False,
            "task_token_use_task_id": False,
            "task_token_from_text": False,
        },
        "tsmoe": {
            "use_moe": True,
            "moe_type": "moe-lora",
            "task_token_enable": False,
            "task_token_use_task_id": False,
            "task_token_from_text": False,
        },
        "tsmoe_tstoken": {
            "use_moe": True,
            "moe_type": "moe-lora",
            "task_token_enable": True,
            "task_token_use_task_id": False,
            "task_token_from_text": True,
        },
    }
    if variant not in presets:
        raise RuntimeError(f"Unsupported model variant: {variant}")
    for attr, value in presets[variant].items():
        if hasattr(exp.model_config, attr):
            setattr(exp.model_config, attr, value)


def build_runtime_from_benchmark(args):
    module = load_module_from_path(args.benchmark_script)
    if not hasattr(module, "Pi0Exp"):
        raise RuntimeError(f"{args.benchmark_script} does not define Pi0Exp")

    exp = module.Pi0Exp()
    apply_model_variant(exp, args.model_variant)

    if args.model_name_or_path:
        exp.model_config.model_name_or_path = args.model_name_or_path
    if args.dataset_name:
        exp.data_config.dataset_name = args.dataset_name

    if args.norm_stats_path:
        exp.data_config.action_config.statistic_mapping = args.norm_stats_path
    else:
        candidate = os.path.join(exp.model_config.model_name_or_path, "norm_stats.json")
        if megfile.smart_exists(candidate):
            exp.data_config.action_config.statistic_mapping = candidate

    if not exp.data_config.action_config.statistic_mapping:
        raise RuntimeError(
            "No norm stats path set. Please pass --norm_stats_path or ensure <model_name_or_path>/norm_stats.json exists."
        )

    for attr, value in (
        ("use_moe", args.use_moe),
        ("moe_type", args.moe_type),
        ("num_experts", args.num_experts),
        ("moe_top_k", args.moe_top_k),
        ("moe_weight_path", args.moe_weight_path),
        ("dense_moe", args.dense_moe),
        ("task_router", args.task_router),
        ("task_router_prior", args.task_router_prior),
        ("task_router_num_tasks", args.task_router_num_tasks),
        ("task_token_enable", args.task_token_enable),
        ("task_token_use_task_id", args.task_token_use_task_id),
        ("task_token_from_text", args.task_token_from_text),
    ):
        if value is not None and hasattr(exp.model_config, attr):
            setattr(exp.model_config, attr, value)

    if args.task_id_from_prompt is not None and hasattr(exp.data_config, "task_id_from_prompt"):
        exp.data_config.task_id_from_prompt = bool(args.task_id_from_prompt)
    if args.report_task_prompts is not None and hasattr(exp.data_config, "report_task_prompts"):
        exp.data_config.report_task_prompts = bool(args.report_task_prompts)

    tokenizer_kwargs = {
        "model_max_length": exp.trainer_config.model_max_length,
        "padding_side": "right",
        "use_fast": exp.tokenizer_config.use_fast_tokenizer,
    }
    tokenizer = exp.tokenizer_config.build_tokenizer(
        exp.model_config.model_name_or_path,
        **tokenizer_kwargs,
    )

    model = exp.model_config.build_model()
    tokenizer = exp.tokenizer_config.add_special_tokens(
        exp.data_config.action_config.string_format,
        exp.data_config.action_config.vocab_size,
        tokenizer,
        model,
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer)

    dataset, _ = exp.data_config.build_data(
        tokenizer,
        exp.model_config.chat_template,
        model.model.mm_vision_module.image_processor,
    )

    return exp, model, dataset, data_collator


def run_analysis(args) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    exp, model, dataset, base_collator = build_runtime_from_benchmark(args)

    dataset, id_to_name = ensure_task_ids(dataset)
    collator = InterpretabilityCollator(base_collator)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    num_workers = max(int(args.num_workers), 0)
    dataloader = DataLoader(
        dataset,
        batch_size=max(int(args.batch_size), 1),
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
    )

    default_pseudo = int(getattr(model.config, "num_experts", args.num_pseudo_experts) or args.num_pseudo_experts)
    num_pseudo_experts = max(int(args.num_pseudo_experts or default_pseudo), 1)

    collector = InterpretabilityCollector(
        model,
        num_pseudo_experts=num_pseudo_experts,
        collect_enabled=args.collect_interpretability,
        save_moe_raw_limit_batches=args.save_moe_raw_limit_batches,
        save_token_routes_limit_batches=args.save_token_routes_limit_batches,
    )

    max_batches = max(int(args.max_batches), 0)

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collect interpretability")):
                if max_batches > 0 and batch_idx >= max_batches:
                    break

                task_ids = batch.get("task_ids")
                if task_ids is None:
                    raise RuntimeError("Batch does not contain task_ids")

                collector.start_batch(task_ids)
                batch = to_device(batch, device)

                model(
                    input_ids=batch.get("input_ids"),
                    attention_mask=batch.get("attention_mask"),
                    actions=batch.get("actions"),
                    states=batch.get("states"),
                    images=batch.get("images"),
                    image_masks=batch.get("image_masks"),
                    task_ids=batch.get("task_ids"),
                    collect_interpretability=args.collect_interpretability,
                )
                collector.end_batch()
    finally:
        collector.close()

    result = collector.finalize()
    task_ids = result["task_ids"]
    counts = result["counts"]
    matrix = result["matrix"]
    router_prob_matrix = result["router_prob_matrix"]
    topk_matrix = result["topk_matrix"]
    layer_top1_matrices = result["layer_top1_matrices"]

    task_labels = _format_task_labels(task_ids, id_to_name)
    metrics = compute_metrics(matrix, counts)

    cluster_order = cluster_task_order(matrix)
    clustered_matrix = matrix[cluster_order]
    clustered_labels = [task_labels[i] for i in cluster_order.tolist()]

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_file = output_dir / "task_expert_matrix.npy"
    np.save(matrix_file, matrix)
    np.save(output_dir / "task_expert_matrix_top1.npy", matrix)

    clustered_file = output_dir / "task_expert_matrix_clustered.npy"
    np.save(clustered_file, clustered_matrix)

    if topk_matrix is not None:
        np.save(output_dir / "task_expert_matrix_topk.npy", topk_matrix)
    if router_prob_matrix is not None:
        np.save(output_dir / "task_expert_matrix_router_prob.npy", router_prob_matrix)

    save_heatmap(
        matrix,
        task_labels,
        str(output_dir / "task_expert_top1_heatmap.png"),
        title=f"Task x Expert Top-1 Usage ({args.model_label})",
    )
    save_heatmap(
        clustered_matrix,
        clustered_labels,
        str(output_dir / "task_expert_top1_clustered_heatmap.png"),
        title=f"Clustered Task x Expert Top-1 Usage ({args.model_label})",
    )
    # Backward-compatible aliases for older plotting/compare scripts.
    save_heatmap(
        matrix,
        task_labels,
        str(output_dir / "task_expert_heatmap.png"),
        title=f"Task x Expert Top-1 Usage ({args.model_label})",
    )
    save_heatmap(
        clustered_matrix,
        clustered_labels,
        str(output_dir / "task_expert_clustered_heatmap.png"),
        title=f"Clustered Task x Expert Top-1 Usage ({args.model_label})",
    )

    layer_metrics = []
    layer_dir = output_dir / "per_layer_top1"
    if layer_top1_matrices:
        layer_dir.mkdir(parents=True, exist_ok=True)
    for layer_record in layer_top1_matrices:
        layer_idx = int(layer_record["layer"])
        layer_task_ids = layer_record["task_ids"]
        layer_counts = layer_record["counts"]
        layer_matrix = layer_record["matrix"]
        layer_labels = _format_task_labels(layer_task_ids, id_to_name)
        np.save(layer_dir / f"layer_{layer_idx:02d}_task_expert_top1.npy", layer_matrix)
        save_heatmap(
            layer_matrix,
            layer_labels,
            str(layer_dir / f"layer_{layer_idx:02d}_task_expert_top1_heatmap.png"),
            title=f"Layer {layer_idx} Task x Expert Top-1 ({args.model_label})",
        )
        layer_metric = compute_metrics(layer_matrix, layer_counts)
        layer_metric["layer"] = layer_idx
        layer_metrics.append(layer_metric)

    bar_files = save_task_barplots(
        matrix,
        task_labels,
        str(output_dir),
        prefix="router" if result["mode"] == "moe" else "pseudo",
        tasks_per_page=args.tasks_per_page,
    )

    pairwise_js = np.asarray(metrics["pairwise_js_matrix"], dtype=np.float64)
    save_pairwise_js_heatmap(pairwise_js, task_labels, str(output_dir / "pairwise_js_heatmap.png"))

    metrics_payload = {
        "analysis_mode": "offline",
        "model_label": args.model_label,
        "mode": result["mode"],
        "task_ids": task_ids,
        "task_labels": task_labels,
        "task_counts": counts.tolist(),
        "cluster_order": cluster_order.tolist(),
        "matrix_semantics": "P(top1_expert | task), averaged over routed tokens and MoE layers",
        "router_prob_matrix_file": "task_expert_matrix_router_prob.npy" if router_prob_matrix is not None else None,
        "topk_matrix_file": "task_expert_matrix_topk.npy" if topk_matrix is not None else None,
        "layer_metrics": layer_metrics,
        **metrics,
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2, ensure_ascii=True)

    if result["raw_records"]:
        np.save(output_dir / "moe_raw_records.npy", np.array(result["raw_records"], dtype=object), allow_pickle=True)
    if result["token_route_records"]:
        np.save(
            output_dir / "token_top1_routes.npy",
            np.array(result["token_route_records"], dtype=object),
            allow_pickle=True,
        )

    summary = {
        "analysis_mode": "offline",
        "output_dir": str(output_dir),
        "num_tasks": int(metrics["num_tasks"]),
        "num_experts": int(metrics["num_experts"]),
        "mode": result["mode"],
        "mean_entropy": float(metrics["mean_entropy"]),
        "weighted_entropy": float(metrics["weighted_entropy"]),
        "normalized_global_entropy": float(metrics["normalized_global_entropy"]),
        "effective_experts": float(metrics["effective_experts"]),
        "max_expert_usage": float(metrics["max_expert_usage"]),
        "pairwise_js_mean": float(metrics["pairwise_js_mean"]),
        "mi_task_expert": float(metrics["mi_task_expert"]),
        "num_layer_heatmaps": len(layer_metrics),
        "bar_plot_files": [str(Path(x).resolve()) for x in bar_files],
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(json.dumps(summary, indent=2, ensure_ascii=True))


def run_compare(args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dirs = [Path(p).resolve() for p in args.compare_run_dirs]
    if not run_dirs:
        raise RuntimeError("--compare_run_dirs is empty")

    labels = args.compare_labels or [d.name for d in run_dirs]
    if len(labels) != len(run_dirs):
        raise RuntimeError("--compare_labels size must match --compare_run_dirs")

    records = []
    for run_dir, label in zip(run_dirs, labels):
        metrics_file = run_dir / "metrics.json"
        matrix_file = run_dir / "task_expert_matrix.npy"
        if not metrics_file.exists() or not matrix_file.exists():
            raise RuntimeError(f"Missing metrics/matrix in {run_dir}")

        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        matrix = np.load(matrix_file)
        order = cluster_task_order(matrix)
        records.append(
            {
                "label": label,
                "metrics": metrics,
                "matrix": matrix,
                "clustered": matrix[order],
            }
        )

    output_dir = Path(args.compare_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, len(records), figsize=(6.0 * len(records), 6.0), dpi=140)
    if len(records) == 1:
        axes = [axes]

    for ax, rec in zip(axes, records):
        mat = rec["clustered"]
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0.0, vmax=float(mat.max(initial=1e-8)))
        ax.set_title(
            (
                f"{rec['label']}\n"
                f"H={rec['metrics']['weighted_entropy']:.4f}, "
                f"JS={rec['metrics']['pairwise_js_mean']:.4f}, "
                f"MI={rec['metrics']['mi_task_expert']:.4f}, "
                f"LB={rec['metrics'].get('normalized_global_entropy', 0.0):.3f}"
            ),
            fontsize=9,
        )
        ax.set_xlabel("Expert")
        ax.set_ylabel("Task (clustered)")
        ax.set_xticks(np.arange(mat.shape[1]))

    fig.colorbar(im, ax=axes, shrink=0.75)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_clustered_heatmaps.png")
    plt.close(fig)

    metric_names = [
        "weighted_entropy",
        "pairwise_js_mean",
        "mi_task_expert",
        "normalized_global_entropy",
        "max_expert_usage",
    ]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 4), dpi=140)
    for i, metric in enumerate(metric_names):
        values = [float(rec["metrics"].get(metric, 0.0)) for rec in records]
        axes[i].bar(np.arange(len(records)), values)
        axes[i].set_xticks(np.arange(len(records)))
        axes[i].set_xticklabels(labels, rotation=20, ha="right")
        axes[i].set_title(metric)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_metrics.png")
    plt.close(fig)

    compare_payload = {
        "runs": [
            {
                "label": rec["label"],
                "weighted_entropy": float(rec["metrics"]["weighted_entropy"]),
                "pairwise_js_mean": float(rec["metrics"]["pairwise_js_mean"]),
                "mi_task_expert": float(rec["metrics"]["mi_task_expert"]),
                "normalized_global_entropy": float(
                    rec["metrics"].get("normalized_global_entropy", 0.0)
                ),
                "effective_experts": float(rec["metrics"].get("effective_experts", 0.0)),
                "max_expert_usage": float(rec["metrics"].get("max_expert_usage", 0.0)),
            }
            for rec in records
        ]
    }
    with open(output_dir / "comparison_metrics.json", "w", encoding="utf-8") as f:
        json.dump(compare_payload, f, indent=2, ensure_ascii=True)

    print(json.dumps(compare_payload, indent=2, ensure_ascii=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task-conditioned expert usage interpretability for LIBERO PI0 models")

    parser.add_argument(
        "--benchmark_script",
        type=str,
        default=str(Path(__file__).resolve().with_name("libero_pi0.py")),
        help="Path to benchmark script defining Pi0Exp",
    )

    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--norm_stats_path", type=str, default=None)
    parser.add_argument(
        "--model_variant",
        type=str,
        choices=["vanilla_moe", "tsmoe", "tsmoe_tstoken"],
        default=None,
        help="Optional preset for the three LIBERO PI0 interpretability variants.",
    )

    parser.add_argument("--output_dir", type=str, default="/mlp_vepfs/share/yjh/workspace/MT-VLA-pi0/interpretability_outputs")
    parser.add_argument("--model_label", type=str, default="model")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=0, help="0 means all batches")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--collect_interpretability", type=parse_bool, default=True)
    parser.add_argument("--num_pseudo_experts", type=int, default=4)
    parser.add_argument("--tasks_per_page", type=int, default=16)
    parser.add_argument("--save_moe_raw_limit_batches", type=int, default=0)
    parser.add_argument(
        "--save_token_routes_limit_batches",
        type=int,
        default=32,
        help="Save per-layer per-token top-1 expert routes for the first N batches; 0 disables raw route dumps.",
    )

    parser.add_argument("--task_id_from_prompt", type=parse_optional_bool, default=None)
    parser.add_argument("--report_task_prompts", type=parse_optional_bool, default=None)

    parser.add_argument("--use_moe", type=parse_optional_bool, default=None)
    parser.add_argument("--moe_type", type=str, default=None)
    parser.add_argument("--num_experts", type=int, default=None)
    parser.add_argument("--moe_top_k", type=int, default=None)
    parser.add_argument("--moe_weight_path", type=str, default=None)
    parser.add_argument("--dense_moe", type=parse_optional_bool, default=None)
    parser.add_argument("--task_router", type=str, default=None)
    parser.add_argument("--task_router_prior", type=parse_optional_bool, default=None)
    parser.add_argument("--task_router_num_tasks", type=int, default=None)

    parser.add_argument("--task_token_enable", type=parse_optional_bool, default=None)
    parser.add_argument("--task_token_use_task_id", type=parse_optional_bool, default=None)
    parser.add_argument("--task_token_from_text", type=parse_optional_bool, default=None)

    parser.add_argument("--compare_run_dirs", type=str, nargs="*", default=None)
    parser.add_argument("--compare_labels", type=str, nargs="*", default=None)
    parser.add_argument("--compare_output_dir", type=str, default="./interpretability_compare")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.compare_run_dirs:
        run_compare(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
