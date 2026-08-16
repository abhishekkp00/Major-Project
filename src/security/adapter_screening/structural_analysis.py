"""
structural_analysis.py
======================
Structural Analysis Module for LoRA Adapter Security Screening.

Analyzes LoRA adapter parameter weights for:
  - Parameter norms (L1, L2, Frobenius norm)
  - Layer-wise parameter statistics (mean, std, min, max, skewness, kurtosis)
  - Rank utilization and singular value spectral statistics
  - Sparsity (ratio of near-zero weights)
  - Outlier layers (Z-score detection across layer norms)
  - Cosine / Frobenius similarity against trusted adapter parameter distributions

Output serves as evidence for downstream risk scoring, not a binary accusation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("secure_lora.security.adapter_screening.structural_analysis")


@dataclass
class LayerStructuralMetrics:
    layer_name: str
    num_params: int
    l1_norm: float
    l2_norm: float
    frobenius_norm: float
    mean: float
    std: float
    min_val: float
    max_val: float
    sparsity_ratio: float
    effective_rank: int
    top_singular_value: float
    spectral_condition_number: float
    z_score: float = 0.0
    is_outlier: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StructuralEvidence:
    total_parameters: int
    layer_count: int
    global_frobenius_norm: float
    global_l2_norm: float
    max_layer_zscore: float
    outlier_layers: List[str] = field(default_factory=list)
    rank_collapse_count: int = 0
    max_condition_number: float = 1.0
    similarity_to_trusted: Optional[float] = None
    layer_metrics: Dict[str, LayerStructuralMetrics] = field(default_factory=dict)
    evidence_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["layer_metrics"] = {k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in self.layer_metrics.items()}
        return d


class StructuralAnalyzer:
    """Performs static structural parameter analysis on candidate LoRA weights."""

    def __init__(
        self,
        near_zero_threshold: float = 1e-6,
        outlier_zscore_threshold: float = 3.0,
        max_condition_number_threshold: float = 1e4,
    ):
        self.near_zero_threshold = near_zero_threshold
        self.outlier_zscore_threshold = outlier_zscore_threshold
        self.max_condition_number_threshold = max_condition_number_threshold

    def analyze(
        self,
        weights: Dict[str, Any],
        trusted_weights: Optional[Dict[str, Any]] = None,
    ) -> StructuralEvidence:
        """Analyzes a dictionary of weight tensors/arrays for structural evidence."""
        if not weights:
            return StructuralEvidence(
                total_parameters=0,
                layer_count=0,
                global_frobenius_norm=0.0,
                global_l2_norm=0.0,
                max_layer_zscore=0.0,
                evidence_notes=["Empty or invalid weights provided."],
            )

        total_params = 0
        layer_metrics_map: Dict[str, LayerStructuralMetrics] = {}
        layer_frob_norms: List[float] = []

        # 1. Compute per-layer statistics
        for name, tensor in weights.items():
            arr = self._to_numpy(tensor)
            if arr is None or arr.size == 0:
                continue

            num_p = arr.size
            total_params += num_p

            l1 = float(np.sum(np.abs(arr)))
            l2 = float(np.linalg.norm(arr.ravel()))
            frob = float(np.linalg.norm(arr))
            layer_frob_norms.append(frob)

            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            min_val = float(np.min(arr))
            max_val = float(np.max(arr))
            sparsity = float(np.mean(np.abs(arr) < self.near_zero_threshold))

            # Spectral statistics if 2D or reshaped to 2D
            eff_rank, top_sv, cond_num = self._compute_spectral_stats(arr)

            layer_metrics_map[name] = LayerStructuralMetrics(
                layer_name=name,
                num_params=num_p,
                l1_norm=round(l1, 4),
                l2_norm=round(l2, 4),
                frobenius_norm=round(frob, 4),
                mean=round(mean_val, 6),
                std=round(std_val, 6),
                min_val=round(min_val, 6),
                max_val=round(max_val, 6),
                sparsity_ratio=round(sparsity, 4),
                effective_rank=eff_rank,
                top_singular_value=round(top_sv, 4),
                spectral_condition_number=round(cond_num, 4),
            )

        if not layer_frob_norms:
            return StructuralEvidence(
                total_parameters=0,
                layer_count=0,
                global_frobenius_norm=0.0,
                global_l2_norm=0.0,
                max_layer_zscore=0.0,
                evidence_notes=["No non-empty parameters found."],
            )

        # 2. Compute Z-scores & Median Norm Ratios for outlier layer detection
        elem_norms = [m.frobenius_norm / np.sqrt(max(1, m.num_params)) for m in layer_metrics_map.values()]
        mean_elem_norm = float(np.mean(elem_norms))
        std_elem_norm = float(np.std(elem_norms))
        median_elem_norm = float(np.median(elem_norms))
        outlier_layers: List[str] = []
        max_zscore = 0.0

        for (name, metrics), elem_n in zip(layer_metrics_map.items(), elem_norms):
            z = float((elem_n - mean_elem_norm) / (std_elem_norm + 1e-8)) if std_elem_norm > 1e-8 else 0.0
            norm_ratio = float(elem_n / (median_elem_norm + 1e-8)) if median_elem_norm > 1e-8 else 1.0
            metrics.z_score = round(z, 4)
            if abs(z) > max_zscore:
                max_zscore = abs(z)
            if abs(z) >= self.outlier_zscore_threshold or norm_ratio >= 4.0:
                metrics.is_outlier = True
                outlier_layers.append(name)

        # 3. Global aggregates
        global_frob = float(np.sqrt(np.sum(np.array(layer_frob_norms) ** 2)))
        global_l2 = float(np.sqrt(sum(m.l2_norm ** 2 for m in layer_metrics_map.values())))
        max_cond = max((m.spectral_condition_number for m in layer_metrics_map.values()), default=1.0)
        rank_collapse_count = sum(1 for m in layer_metrics_map.values() if m.effective_rank <= 1)

        # 4. Compare with trusted weights if provided
        similarity_to_trusted = None
        if trusted_weights:
            similarity_to_trusted = self._compute_similarity(weights, trusted_weights)

        # 5. Build evidence notes
        notes = [
            f"Analyzed {len(layer_metrics_map)} layers containing {total_params} parameters.",
            f"Global Frobenius norm: {global_frob:.4f}, max layer Z-score: {max_zscore:.2f}.",
        ]
        if outlier_layers:
            notes.append(f"Flagged {len(outlier_layers)} outlier layer(s) exceeding Z-score threshold {self.outlier_zscore_threshold}.")
        if rank_collapse_count > 0:
            notes.append(f"Detected {rank_collapse_count} layer(s) with effective rank <= 1.")
        if max_cond > self.max_condition_number_threshold:
            notes.append(f"High spectral condition number detected: {max_cond:.1f}.")
        if similarity_to_trusted is not None:
            notes.append(f"Cosine similarity to trusted adapter distribution: {similarity_to_trusted:.4f}.")

        return StructuralEvidence(
            total_parameters=total_params,
            layer_count=len(layer_metrics_map),
            global_frobenius_norm=round(global_frob, 4),
            global_l2_norm=round(global_l2, 4),
            max_layer_zscore=round(max_zscore, 4),
            outlier_layers=outlier_layers,
            rank_collapse_count=rank_collapse_count,
            max_condition_number=round(max_cond, 4),
            similarity_to_trusted=round(similarity_to_trusted, 4) if similarity_to_trusted is not None else None,
            layer_metrics=layer_metrics_map,
            evidence_notes=notes,
        )

    def _to_numpy(self, tensor: Any) -> Optional[np.ndarray]:
        """Converts PyTorch Tensor or NumPy array to 2D/N-D NumPy float array."""
        try:
            if hasattr(tensor, "detach"):
                arr = tensor.detach().cpu().numpy()
            elif isinstance(tensor, np.ndarray):
                arr = tensor
            else:
                arr = np.array(tensor, dtype=np.float32)
            return arr.astype(np.float32)
        except Exception as e:
            logger.warning("Could not convert weight to numpy: %s", e)
            return None

    def _compute_spectral_stats(self, arr: np.ndarray) -> Tuple[int, float, float]:
        """Computes effective numerical rank, top singular value, and condition number."""
        if arr.ndim < 2:
            matrix = arr.reshape(1, -1)
        elif arr.ndim > 2:
            matrix = arr.reshape(arr.shape[0], -1)
        else:
            matrix = arr

        try:
            sv = np.linalg.svd(matrix, compute_uv=False)
            if len(sv) == 0:
                return 1, 0.0, 1.0
            top_sv = float(sv[0])
            min_sv = float(sv[-1]) if sv[-1] > 1e-12 else 1e-12
            cond_num = float(top_sv / min_sv)
            eff_rank = int(np.sum(sv > 1e-4 * top_sv))
            return max(1, eff_rank), top_sv, min(cond_num, 1e6)
        except Exception:
            return 1, float(np.max(np.abs(matrix))), 1.0

    def _compute_similarity(self, candidate: Dict[str, Any], trusted: Dict[str, Any]) -> float:
        """Computes average cosine similarity across common parameter layers."""
        sims = []
        for k in candidate:
            if k in trusted:
                c_arr = self._to_numpy(candidate[k]).ravel()
                t_arr = self._to_numpy(trusted[k]).ravel()
                if c_arr.size == t_arr.size and c_arr.size > 0:
                    norm_c = np.linalg.norm(c_arr)
                    norm_t = np.linalg.norm(t_arr)
                    if norm_c > 1e-12 and norm_t > 1e-12:
                        cos_sim = float(np.dot(c_arr, t_arr) / (norm_c * norm_t))
                        sims.append(cos_sim)

        return float(np.mean(sims)) if sims else 1.0
