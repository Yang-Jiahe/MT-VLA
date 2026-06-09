from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812


def _task_label(value: Any) -> str:
    text = str(value).strip()
    return text or "__empty_task__"


def _compute_pairwise_js(matrix: np.ndarray) -> np.ndarray:
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
            value = 0.5 * (kl_pm + kl_qm)
            js[i, j] = value
            js[j, i] = value
    return js


def _compute_metrics(matrix: np.ndarray, counts: np.ndarray) -> dict[str, Any]:
    eps = 1e-12
    entropy_per_task = -np.sum(matrix * np.log(matrix + eps), axis=1)
    weighted_entropy = float(
        np.sum((counts / np.clip(counts.sum(), eps, None)) * entropy_per_task)
    )

    pairwise_js = _compute_pairwise_js(matrix)
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

    global_usage = row_mass.sum(axis=0)
    global_usage = global_usage / np.clip(global_usage.sum(), a_min=eps, a_max=None)
    global_entropy = -float(np.sum(global_usage * np.log(global_usage + eps)))
    normalized_global_entropy = (
        global_entropy / float(np.log(matrix.shape[1])) if matrix.shape[1] > 1 else 1.0
    )

    return {
        "num_tasks": int(matrix.shape[0]),
        "num_experts": int(matrix.shape[1]),
        "mean_entropy": float(entropy_per_task.mean()),
        "weighted_entropy": weighted_entropy,
        "pairwise_js_mean": pairwise_js_mean,
        "mi_task_expert": float(mi),
        "global_usage": global_usage.tolist(),
        "global_entropy": global_entropy,
        "normalized_global_entropy": float(normalized_global_entropy),
        "effective_experts": float(np.exp(global_entropy)),
        "max_expert_usage": float(global_usage.max()),
        "min_expert_usage": float(global_usage.min()),
        "entropy_per_task": entropy_per_task.tolist(),
        "pairwise_js_matrix": pairwise_js.tolist(),
    }


def _cluster_task_order(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] <= 2:
        return np.arange(matrix.shape[0], dtype=np.int64)
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        from scipy.spatial.distance import pdist

        dist = pdist(matrix, metric="jensenshannon")
        dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
        if float(dist.max()) <= 0.0:
            return np.arange(matrix.shape[0], dtype=np.int64)
        return leaves_list(linkage(dist, method="average")).astype(np.int64)
    except Exception:
        entropy = -np.sum(matrix * np.log(np.clip(matrix, 1e-12, None)), axis=1)
        primary = np.argmax(matrix, axis=1)
        return np.lexsort((entropy, primary)).astype(np.int64)


def _save_heatmap(
    matrix: np.ndarray,
    task_labels: list[str],
    out_file: str,
    *,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    num_tasks, num_experts = matrix.shape
    fig_w = max(7.0, 1.2 * num_experts)
    fig_h = max(5.0, 0.28 * num_tasks)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=140)
    image = ax.imshow(matrix, aspect="auto", cmap="magma", vmin=0.0, vmax=float(matrix.max()))
    ax.set_title(title)
    ax.set_xlabel("Expert")
    ax.set_ylabel("Task")
    ax.set_xticks(np.arange(num_experts))
    ax.set_xticklabels([f"E{i}" for i in range(num_experts)])
    if num_tasks <= 60:
        ax.set_yticks(np.arange(num_tasks))
        ax.set_yticklabels(task_labels, fontsize=7)
    else:
        step = max(1, num_tasks // 40)
        ticks = np.arange(0, num_tasks, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([task_labels[i] for i in ticks], fontsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("P(top1 expert | task)")
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)


class Pi0OnlineRouteCollector:
    """Collects task-conditioned MoE routing during online inference rollouts."""

    def __init__(
        self,
        model,
        *,
        output_dir: str,
        model_label: str = "online",
        flush_steps: int = 10,
        save_token_routes_limit_batches: int = 32,
        save_moe_raw_limit_batches: int = 0,
    ):
        self.model = model
        self.output_dir = Path(output_dir).resolve()
        self.model_label = str(model_label)
        self.flush_steps = max(int(flush_steps), 1)
        self.save_token_routes_limit_batches = max(int(save_token_routes_limit_batches), 0)
        self.save_moe_raw_limit_batches = max(int(save_moe_raw_limit_batches), 0)

        self.hooks = []
        self.mode: Optional[str] = None
        self.request_index = 0
        self.current_task_labels: Optional[list[str]] = None
        self.current_layer_indices: list[int] = []
        self.current_router_by_layer: list[np.ndarray] = []
        self.current_top1_by_layer: list[np.ndarray] = []
        self.current_topk_by_layer: list[np.ndarray] = []

        self.task_router_sum: dict[str, np.ndarray] = {}
        self.task_top1_sum: dict[str, np.ndarray] = {}
        self.task_topk_sum: dict[str, np.ndarray] = {}
        self.task_counts: dict[str, int] = defaultdict(int)
        self.task_layer_top1_sum: dict[int, dict[str, np.ndarray]] = {}
        self.task_layer_counts: dict[int, dict[str, int]] = {}

        self.token_route_records: list[dict[str, Any]] = []
        self.raw_records: list[dict[str, Any]] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        action_expert = getattr(getattr(self.model, "model", None), "action_expert", None)
        layers = getattr(action_expert, "layers", None)
        if layers is None:
            return
        for layer_idx, layer in enumerate(layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue

            def _hook(module, inputs, _layer_idx=layer_idx):
                self._on_mlp_forward_pre(_layer_idx, module, inputs)

            self.hooks.append(mlp.register_forward_pre_hook(_hook))

    def close(self) -> None:
        self.save()
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def reset(self) -> None:
        self.request_index = 0
        self.current_task_labels = None
        self.current_layer_indices = []
        self.current_router_by_layer = []
        self.current_top1_by_layer = []
        self.current_topk_by_layer = []
        self.task_router_sum.clear()
        self.task_top1_sum.clear()
        self.task_topk_sum.clear()
        self.task_counts.clear()
        self.task_layer_top1_sum.clear()
        self.task_layer_counts.clear()
        self.token_route_records = []
        self.raw_records = []

    def start_batch(self, task_labels: list[str]) -> None:
        self.current_task_labels = [_task_label(label) for label in task_labels]
        self.current_layer_indices = []
        self.current_router_by_layer = []
        self.current_top1_by_layer = []
        self.current_topk_by_layer = []

    def end_batch(self) -> None:
        if self.current_task_labels is None:
            return
        if not self.current_top1_by_layer:
            self.request_index += 1
            self.current_task_labels = None
            return

        router_usage = np.mean(np.stack(self.current_router_by_layer, axis=0), axis=0)
        top1_usage = np.mean(np.stack(self.current_top1_by_layer, axis=0), axis=0)
        topk_usage = np.mean(np.stack(self.current_topk_by_layer, axis=0), axis=0)

        for idx, task_label in enumerate(self.current_task_labels):
            if task_label not in self.task_router_sum:
                self.task_router_sum[task_label] = np.zeros(router_usage.shape[-1], dtype=np.float64)
            if task_label not in self.task_top1_sum:
                self.task_top1_sum[task_label] = np.zeros(top1_usage.shape[-1], dtype=np.float64)
            if task_label not in self.task_topk_sum:
                self.task_topk_sum[task_label] = np.zeros(topk_usage.shape[-1], dtype=np.float64)
            self.task_router_sum[task_label] += router_usage[idx].astype(np.float64)
            self.task_top1_sum[task_label] += top1_usage[idx].astype(np.float64)
            self.task_topk_sum[task_label] += topk_usage[idx].astype(np.float64)
            self.task_counts[task_label] += 1

        for layer_idx, layer_top1_usage in zip(
            self.current_layer_indices, self.current_top1_by_layer
        ):
            layer_sum = self.task_layer_top1_sum.setdefault(int(layer_idx), {})
            layer_counts = self.task_layer_counts.setdefault(int(layer_idx), {})
            for idx, task_label in enumerate(self.current_task_labels):
                if task_label not in layer_sum:
                    layer_sum[task_label] = np.zeros(
                        layer_top1_usage.shape[-1], dtype=np.float64
                    )
                layer_sum[task_label] += layer_top1_usage[idx].astype(np.float64)
                layer_counts[task_label] = layer_counts.get(task_label, 0) + 1

        self.request_index += 1
        self.current_task_labels = None

    def maybe_flush(self) -> None:
        if self.request_index > 0 and self.request_index % self.flush_steps == 0:
            self.save()

    @staticmethod
    def _is_moe_module(module) -> bool:
        return bool(getattr(module, "use_moe", False)) and getattr(module, "moe_ffn", None) is not None

    def _on_mlp_forward_pre(self, layer_idx: int, module, inputs) -> None:
        if self.current_task_labels is None:
            return
        if not inputs:
            return
        x = inputs[0]
        if not torch.is_tensor(x) or not self._is_moe_module(module):
            return
        with torch.no_grad():
            router_usage, top1_usage, topk_usage, raw_router, raw_top1, raw_topk = (
                self._collect_moe_usage(module, x)
            )
        self.mode = "moe"
        self.current_layer_indices.append(layer_idx)
        self.current_router_by_layer.append(router_usage)
        self.current_top1_by_layer.append(top1_usage)
        self.current_topk_by_layer.append(topk_usage)
        if self.request_index < self.save_token_routes_limit_batches:
            self.token_route_records.append(
                {
                    "request": int(self.request_index),
                    "layer": int(layer_idx),
                    "task_labels": list(self.current_task_labels),
                    "top1_idx": raw_top1,
                }
            )
        if self.request_index < self.save_moe_raw_limit_batches:
            self.raw_records.append(
                {
                    "request": int(self.request_index),
                    "layer": int(layer_idx),
                    "task_labels": list(self.current_task_labels),
                    "router_probs": raw_router,
                    "top1_idx": raw_top1,
                    "topk_idx": raw_topk,
                }
            )

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
                raise RuntimeError("MoE FFN does not expose w_gating or _routing")
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

        top1_usage = F.one_hot(top1_idx, num_experts).float().mean(dim=1)
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

    @staticmethod
    def _build_matrix_from_sum(
        task_sum: dict[str, np.ndarray],
        task_counts: dict[str, int],
    ) -> tuple[list[str], np.ndarray, np.ndarray]:
        task_labels = sorted(task_sum.keys())
        raw_mass = []
        counts = []
        labels = []
        for label in task_labels:
            count = int(task_counts.get(label, 0))
            if count <= 0:
                continue
            raw_mass.append(task_sum[label] / float(count))
            counts.append(count)
            labels.append(label)
        if not raw_mass:
            raise RuntimeError("No online interpretability samples were collected.")
        matrix = np.asarray(raw_mass, dtype=np.float64)
        matrix = matrix / np.clip(matrix.sum(axis=1, keepdims=True), 1e-12, None)
        return labels, np.asarray(counts, dtype=np.int64), matrix

    def save(self) -> dict[str, Any]:
        if not self.task_top1_sum:
            return {
                "analysis_mode": "online",
                "model_label": self.model_label,
                "num_requests": int(self.request_index),
                "status": "empty",
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        task_labels, counts, matrix = self._build_matrix_from_sum(
            self.task_top1_sum, self.task_counts
        )
        metrics = _compute_metrics(matrix, counts)
        order = _cluster_task_order(matrix)
        clustered_matrix = matrix[order]
        clustered_labels = [task_labels[i] for i in order.tolist()]

        np.save(self.output_dir / "online_task_expert_matrix.npy", matrix)
        np.save(self.output_dir / "online_task_expert_matrix_top1.npy", matrix)
        np.save(self.output_dir / "online_task_expert_matrix_clustered.npy", clustered_matrix)
        if self.task_router_sum:
            _, _, router_matrix = self._build_matrix_from_sum(
                self.task_router_sum, self.task_counts
            )
            np.save(self.output_dir / "online_task_expert_matrix_router_prob.npy", router_matrix)
        if self.task_topk_sum:
            _, _, topk_matrix = self._build_matrix_from_sum(
                self.task_topk_sum, self.task_counts
            )
            np.save(self.output_dir / "online_task_expert_matrix_topk.npy", topk_matrix)

        _save_heatmap(
            matrix,
            task_labels,
            str(self.output_dir / "online_task_expert_top1_heatmap.png"),
            title=f"Online Task x Expert Top-1 Usage ({self.model_label})",
        )
        _save_heatmap(
            clustered_matrix,
            clustered_labels,
            str(self.output_dir / "online_task_expert_top1_clustered_heatmap.png"),
            title=f"Online Clustered Task x Expert Top-1 ({self.model_label})",
        )

        layer_metrics = []
        layer_dir = self.output_dir / "per_layer_top1"
        layer_dir.mkdir(parents=True, exist_ok=True)
        for layer_idx in sorted(self.task_layer_top1_sum.keys()):
            layer_labels, layer_counts, layer_matrix = self._build_matrix_from_sum(
                self.task_layer_top1_sum[layer_idx],
                self.task_layer_counts.get(layer_idx, {}),
            )
            np.save(layer_dir / f"layer_{layer_idx:02d}_online_task_expert_top1.npy", layer_matrix)
            _save_heatmap(
                layer_matrix,
                layer_labels,
                str(layer_dir / f"layer_{layer_idx:02d}_online_task_expert_top1_heatmap.png"),
                title=f"Online Layer {layer_idx} Task x Expert Top-1 ({self.model_label})",
            )
            layer_metric = _compute_metrics(layer_matrix, layer_counts)
            layer_metric["layer"] = int(layer_idx)
            layer_metrics.append(layer_metric)

        if self.token_route_records:
            np.save(
                self.output_dir / "online_token_top1_routes.npy",
                np.array(self.token_route_records, dtype=object),
                allow_pickle=True,
            )
        if self.raw_records:
            np.save(
                self.output_dir / "online_moe_raw_records.npy",
                np.array(self.raw_records, dtype=object),
                allow_pickle=True,
            )

        payload = {
            "analysis_mode": "online",
            "model_label": self.model_label,
            "num_requests": int(self.request_index),
            "task_labels": task_labels,
            "task_counts": counts.tolist(),
            "cluster_order": order.tolist(),
            "matrix_semantics": (
                "P(top1_expert | task), accumulated from LIBERO rollout "
                "inference_action calls"
            ),
            "layer_metrics": layer_metrics,
            **metrics,
        }
        with open(self.output_dir / "online_metrics.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)

        summary = {
            "analysis_mode": "online",
            "model_label": self.model_label,
            "output_dir": str(self.output_dir),
            "num_requests": int(self.request_index),
            "num_tasks": int(metrics["num_tasks"]),
            "num_experts": int(metrics["num_experts"]),
            "weighted_entropy": float(metrics["weighted_entropy"]),
            "normalized_global_entropy": float(metrics["normalized_global_entropy"]),
            "effective_experts": float(metrics["effective_experts"]),
            "max_expert_usage": float(metrics["max_expert_usage"]),
            "pairwise_js_mean": float(metrics["pairwise_js_mean"]),
            "mi_task_expert": float(metrics["mi_task_expert"]),
            "num_layer_heatmaps": len(layer_metrics),
            "status": "ok",
        }
        with open(self.output_dir / "online_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=True)
        return summary
