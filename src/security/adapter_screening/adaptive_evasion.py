"""
adaptive_evasion.py
===================
Adaptive Adversarial Evasion Benchmark for LoRA Adapter Security Screening.

Defines safe, synthetic, research-bounded adapter generators and optimization objectives
to evaluate detector robustness against adaptive evasion attempts.

Adapter Categories:
  - CLEAN: Standard benign LoRA adapter.
  - BASIC_SUSPICIOUS: Unconstrained synthetic suspicious adapter with obvious structural Z-score anomalies.
  - ADAPTIVE_SUSPICIOUS: Bounded synthetic suspicious adapter constrained to minimize structural distance
                        and maintain high cosine similarity relative to trusted weights while preserving
                        controlled trigger-conditioned behavior.

Evasion Levels:
  - LEVEL 0: Unconstrained structural anomaly (Z >= 15.0, low similarity).
  - LEVEL 1: Lightly constrained structural anomaly (Z ~ 4.0, similarity ~ 0.80).
  - LEVEL 2: Moderately constrained structural anomaly (Z ~ 2.5, similarity ~ 0.92).
  - LEVEL 3: Strongly constrained adaptive anomaly (Z < 1.4, similarity > 0.96).

SAFE RESEARCH NOTICE:
  Contains only synthetic, non-harmful research probes.
  No real malware, no external exploits, no security boundary bypasses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.security.adapter_screening.structural_analysis import StructuralAnalyzer, StructuralEvidence
from src.security.adapter_screening.behavioral_analysis import BehavioralAnalyzer, BehavioralEvidence
from src.security.adapter_screening.risk_scoring import RiskScorer, RiskAssessment, ScreeningThresholdConfig

logger = logging.getLogger("secure_lora.security.adapter_screening.adaptive_evasion")


@dataclass
class StructuralDistanceMetrics:
    norm_distance: float
    distribution_distance: float
    layer_distance: float
    outlier_distance: float
    overall_structural_distance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class AdaptiveAdapterSample:
    sample_id: str
    category: str  # "CLEAN", "BASIC_SUSPICIOUS", "ADAPTIVE_SUSPICIOUS"
    evasion_level: int  # 0, 1, 2, 3
    weights: Dict[str, np.ndarray]
    force_trigger_activation: bool
    ground_truth_anomalous: bool
    structural_distance: StructuralDistanceMetrics
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_structural_distance(
    candidate_weights: Dict[str, np.ndarray],
    trusted_weights: Dict[str, np.ndarray],
) -> StructuralDistanceMetrics:
    """Computes detailed feature-level structural distance between candidate and trusted weights."""
    common_keys = [k for k in candidate_weights if k in trusted_weights]
    if not common_keys:
        return StructuralDistanceMetrics(
            norm_distance=1.0,
            distribution_distance=1.0,
            layer_distance=1.0,
            outlier_distance=1.0,
            overall_structural_distance=1.0,
        )

    cand_flats = [candidate_weights[k].flatten() for k in common_keys]
    trus_flats = [trusted_weights[k].flatten() for k in common_keys]

    cand_vec = np.concatenate(cand_flats)
    trus_vec = np.concatenate(trus_flats)

    # 1. Norm distance (relative Frobenius norm difference)
    norm_c = float(np.linalg.norm(cand_vec))
    norm_t = float(np.linalg.norm(trus_vec))
    norm_dist = float(abs(norm_c - norm_t) / (norm_t + 1e-8))

    # 2. Distribution distance (Wasserstein-1 approximation / mean-std shift)
    mean_shift = float(abs(np.mean(cand_vec) - np.mean(trus_vec)))
    std_shift = float(abs(np.std(cand_vec) - np.std(trus_vec)))
    dist_dist = mean_shift + std_shift

    # 3. Layer distance (mean 1 - cosine similarity per layer)
    layer_cos_sims = []
    for k in common_keys:
        c_k = candidate_weights[k].flatten()
        t_k = trusted_weights[k].flatten()
        nc = np.linalg.norm(c_k)
        nt = np.linalg.norm(t_k)
        if nc > 1e-8 and nt > 1e-8:
            cos_s = float(np.dot(c_k, t_k) / (nc * nt))
            layer_cos_sims.append(cos_s)
    layer_dist = float(1.0 - np.mean(layer_cos_sims)) if layer_cos_sims else 0.0

    # 4. Outlier distance (Z-score difference of max layer norm)
    c_layer_norms = [float(np.linalg.norm(v)) for v in cand_flats]
    t_layer_norms = [float(np.linalg.norm(v)) for v in trus_flats]
    max_c_z = float((max(c_layer_norms) - np.mean(c_layer_norms)) / (np.std(c_layer_norms) + 1e-8))
    max_t_z = float((max(t_layer_norms) - np.mean(t_layer_norms)) / (np.std(t_layer_norms) + 1e-8))
    outlier_dist = float(abs(max_c_z - max_t_z))

    # 5. Composite structural distance
    overall = float(0.30 * norm_dist + 0.20 * dist_dist + 0.35 * layer_dist + 0.15 * min(1.0, outlier_dist / 5.0))

    return StructuralDistanceMetrics(
        norm_distance=round(norm_dist, 4),
        distribution_distance=round(dist_dist, 4),
        layer_distance=round(layer_dist, 4),
        outlier_distance=round(outlier_dist, 4),
        overall_structural_distance=round(overall, 4),
    )


class AdaptiveAdapterFactory:
    """Generates synthetic clean, basic suspicious, and adaptive suspicious adapters for research benchmarks."""

    def __init__(self, rank: int = 8, hidden_dim: int = 64, num_layers: int = 4):
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def generate_clean_adapter(self, seed: int = 42) -> Dict[str, np.ndarray]:
        """Generates clean reference Gaussian LoRA weights."""
        rng = np.random.RandomState(seed)
        weights = {}
        for i in range(self.num_layers):
            weights[f"layer_{i}.lora_A.weight"] = rng.normal(0.0, 0.02, (self.rank, self.hidden_dim)).astype(np.float32)
            weights[f"layer_{i}.lora_B.weight"] = rng.normal(0.0, 0.001, (self.hidden_dim, self.rank)).astype(np.float32)
        return weights

    def generate_basic_suspicious_adapter(self, trusted_weights: Dict[str, np.ndarray], seed: int = 43) -> Dict[str, np.ndarray]:
        """Level 0: Unconstrained suspicious adapter with obvious structural outlier injection."""
        weights = {k: v.copy() for k, v in trusted_weights.items()}
        # Inject massive multiplier on layer 1 (Z >= 15.0)
        outlier_key = "layer_1.lora_A.weight"
        if outlier_key in weights:
            weights[outlier_key] = weights[outlier_key] * 20.0
        return weights

    def generate_adaptive_suspicious_adapter(
        self,
        trusted_weights: Dict[str, np.ndarray],
        evasion_level: int = 1,
        seed: int = 44,
    ) -> Dict[str, np.ndarray]:
        """
        Levels 1, 2, 3: Constrains structural perturbations to remain close to trusted weights.

        Level 1 (Light Constraint): Z ~ 4.0, layer multiplier ~ 3.5x
        Level 2 (Moderate Constraint): Z ~ 2.2, layer multiplier ~ 1.8x, distributed noise
        Level 3 (Strong Constraint): Z < 1.3, cosine sim > 0.97, subtle rank-1 perturbation
        """
        rng = np.random.RandomState(seed)
        weights = {k: v.copy() for k, v in trusted_weights.items()}

        if evasion_level == 1:
            # Mild layer multiplier (Z ~ 4.0)
            if "layer_1.lora_A.weight" in weights:
                weights["layer_1.lora_A.weight"] = weights["layer_1.lora_A.weight"] * 3.5
        elif evasion_level == 2:
            # Subtle perturbation spread across layers (Z ~ 2.2)
            for k in weights:
                weights[k] += rng.normal(0.0, 0.003, weights[k].shape).astype(np.float32)
            if "layer_1.lora_A.weight" in weights:
                weights["layer_1.lora_A.weight"] = weights["layer_1.lora_A.weight"] * 1.6
        elif evasion_level == 3:
            # High-precision orthogonal rank-1 perturbation (Z < 1.3, Similarity > 0.97)
            for k in weights:
                if "lora_A" in k:
                    u = rng.normal(0.0, 0.0005, (weights[k].shape[0], 1)).astype(np.float32)
                    v = rng.normal(0.0, 0.0005, (1, weights[k].shape[1])).astype(np.float32)
                    weights[k] += u @ v
        else:
            raise ValueError(f"Invalid evasion level: {evasion_level}")

        return weights

    def build_benchmark_suite(self, num_samples_per_cat: int = 10, seed: int = 42) -> List[AdaptiveAdapterSample]:
        """Constructs a complete synthetic evaluation suite across Clean, Basic, and Levels 0-3 Adaptive samples."""
        trusted = self.generate_clean_adapter(seed=seed)
        samples: List[AdaptiveAdapterSample] = []

        # 1. Clean Samples (Ground Truth: Clean / False)
        for i in range(num_samples_per_cat):
            s_seed = seed + 100 + i
            w = {k: v + np.random.RandomState(s_seed).normal(0.0, 0.0002, v.shape).astype(np.float32) for k, v in trusted.items()}
            dist = compute_structural_distance(w, trusted)
            samples.append(AdaptiveAdapterSample(
                sample_id=f"clean_{i}",
                category="CLEAN",
                evasion_level=0,
                weights=w,
                force_trigger_activation=False,
                ground_truth_anomalous=False,
                structural_distance=dist,
                metadata={"seed": s_seed, "description": "Clean benign LoRA adapter"},
            ))

        # 2. Basic Suspicious Samples (Level 0, Ground Truth: Anomalous / True)
        for i in range(num_samples_per_cat):
            s_seed = seed + 200 + i
            w = self.generate_basic_suspicious_adapter(trusted, seed=s_seed)
            dist = compute_structural_distance(w, trusted)
            samples.append(AdaptiveAdapterSample(
                sample_id=f"basic_suspicious_{i}",
                category="BASIC_SUSPICIOUS",
                evasion_level=0,
                weights=w,
                force_trigger_activation=True,
                ground_truth_anomalous=True,
                structural_distance=dist,
                metadata={"seed": s_seed, "description": "Unconstrained structural outlier + trigger behavior"},
            ))

        # 3. Adaptive Suspicious Samples - Level 1 (Ground Truth: Anomalous / True)
        for i in range(num_samples_per_cat):
            s_seed = seed + 300 + i
            w = self.generate_adaptive_suspicious_adapter(trusted, evasion_level=1, seed=s_seed)
            dist = compute_structural_distance(w, trusted)
            samples.append(AdaptiveAdapterSample(
                sample_id=f"adaptive_lvl1_{i}",
                category="ADAPTIVE_SUSPICIOUS",
                evasion_level=1,
                weights=w,
                force_trigger_activation=True,
                ground_truth_anomalous=True,
                structural_distance=dist,
                metadata={"seed": s_seed, "description": "Level 1 lightly constrained adaptive adapter"},
            ))

        # 4. Adaptive Suspicious Samples - Level 2 (Ground Truth: Anomalous / True)
        for i in range(num_samples_per_cat):
            s_seed = seed + 400 + i
            w = self.generate_adaptive_suspicious_adapter(trusted, evasion_level=2, seed=s_seed)
            dist = compute_structural_distance(w, trusted)
            samples.append(AdaptiveAdapterSample(
                sample_id=f"adaptive_lvl2_{i}",
                category="ADAPTIVE_SUSPICIOUS",
                evasion_level=2,
                weights=w,
                force_trigger_activation=True,
                ground_truth_anomalous=True,
                structural_distance=dist,
                metadata={"seed": s_seed, "description": "Level 2 moderately constrained adaptive adapter"},
            ))

        # 5. Adaptive Suspicious Samples - Level 3 (Ground Truth: Anomalous / True)
        for i in range(num_samples_per_cat):
            s_seed = seed + 500 + i
            w = self.generate_adaptive_suspicious_adapter(trusted, evasion_level=3, seed=s_seed)
            dist = compute_structural_distance(w, trusted)
            samples.append(AdaptiveAdapterSample(
                sample_id=f"adaptive_lvl3_{i}",
                category="ADAPTIVE_SUSPICIOUS",
                evasion_level=3,
                weights=w,
                force_trigger_activation=True,
                ground_truth_anomalous=True,
                structural_distance=dist,
                metadata={"seed": s_seed, "description": "Level 3 strongly constrained adaptive adapter"},
            ))

        return samples
