"""
pii_metrics.py
==============
Faithful PII/PHI detection and masking evaluation for the Secure Device-Bound
LoRA Fine-Tuning Framework.

DOCUMENTATION & METHODOLOGY:
----------------------------
1. Corpus Size:
   33 labeled test samples (18 True Positive samples containing target PII/PHI entities
   and 15 True Negative samples containing clean control text).

2. PII Classes Evaluated:
   - SSN (Social Security Numbers)
   - EMAIL (RFC 5322 Email addresses)
   - PHONE (ITU-T E.164 & national phone numbers)
   - IP_ADDRESS (IPv4 / IPv6 addresses)
   - API_KEY (API keys, secrets, tokens, passwords)
   - CREDIT_CARD (ISO/IEC 7812 credit cards validated with Luhn algorithm)

3. Ground-Truth Construction:
   Curated ground-truth corpus (`GROUND_TRUTH_CORPUS`) where each sample consists of
   a text string and an explicit set of expected PII entity classes (`expected_types`).

4. Evaluation Methodology:
   Ground-truth text samples are processed directly through the framework's production
   PII engine (`detect_pii_advanced` / `mask_pii_advanced`). Predictions are evaluated
   against ground-truth expected labels to calculate per-class True Positives (TP),
   False Positives (FP), False Negatives (FN), True Negatives (TN), Precision, Recall,
   and F1-score, as well as micro- and macro-averaged metrics.

5. Production Detector Entry Point:
   `src.security.pii_engine.detect_pii_advanced` / `src.security.pii_engine.mask_pii_advanced`
   (invoking `src.security.pii_engine.HybridPIIEngine`). No local or second independent
   detector regexes are used for evaluation.

6. Scope Disclaimer:
   This benchmark evaluates input preprocessing PII/PHI detection and masking performance
   against a labeled ground-truth corpus. It does NOT measure live LLM memorization,
   training data extraction, or model generation leakage.

Usage:
    python -m src.evaluation.pii_metrics
    python -m src.evaluation.pii_metrics --output outputs/benchmarks/pii_metrics.json
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Set

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import re
# --------------------------------------------------------------------------
# Production Detector Entry Points
# --------------------------------------------------------------------------
# Predictions come ONLY from the production HybridPIIEngine.
from src.security.pii_engine import detect_pii_advanced, mask_pii_advanced, HybridPIIEngine, ENTITIES_PATTERNS


def detect_pii(text: str) -> Dict[str, List[str]]:
    """Returns dict of detected PII type -> list of matched strings using the production Hybrid PII Engine."""
    return detect_pii_advanced(text)


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Applies masking to all detected PII using the production Hybrid PII Engine. Returns (masked_text, counts_per_type)."""
    return mask_pii_advanced(text)


# --------------------------------------------------------------------------
# Labeled Ground-Truth Test Corpus (Evaluation Source of Truth)
# --------------------------------------------------------------------------
# Each entry: (text, expected_pii_types_present: Set[str])
# Expected labels come ONLY from this ground-truth corpus.

GROUND_TRUTH_CORPUS: List[Tuple[str, Set[str]]] = [
    # ── TRUE POSITIVES (PII is present) ────────────────────────────────────
    # SSN
    ("Patient John Doe, SSN: 123-45-6789, admitted on Monday.", {"SSN"}),
    ("Employee record: SS Number 987-65-4321 on file.", {"SSN"}),
    ("The file shows SSN 000-12-3456 under the tax section.", {"SSN"}),
    ("Verify identity with 456-78-9012 for enrollment.", {"SSN"}),
    ("SSN mismatch detected for 321-54-9870.", {"SSN"}),

    # EMAIL
    ("Contact the admin at support@example.com for help.", {"EMAIL"}),
    ("Forward results to alice.jones@university.edu immediately.", {"EMAIL"}),
    ("CC: bob+filter@company.co.uk; charlie@test.io", {"EMAIL"}),
    ("Reply-To: noreply@secure-lora.dev", {"EMAIL"}),
    ("User email: john.doe99@gmail.com registered.", {"EMAIL"}),

    # PHONE
    ("Call our hotline at (800) 555-1234 for support.", {"PHONE"}),
    ("Reach Dr. Smith at 415-987-6543.", {"PHONE"}),
    ("Emergency contact: +1 212 555 9876", {"PHONE"}),
    ("Fax: 1-800-555-0101", {"PHONE"}),
    ("Mobile: 650.555.7890", {"PHONE"}),

    # IP_ADDRESS
    ("Server located at 192.168.1.100 is unreachable.", {"IP_ADDRESS"}),
    ("Block traffic from 10.0.0.1 in the firewall rules.", {"IP_ADDRESS"}),
    ("Connection from 203.0.113.42 logged.", {"IP_ADDRESS"}),
    ("Whitelist 172.16.0.5 for internal access.", {"IP_ADDRESS"}),
    ("Ping 8.8.8.8 to check DNS.", {"IP_ADDRESS"}),

    # API_KEY / PASSWORD / SECRET
    ("api_key = 'sk-abc123XYZ789def456ghi'", {"API_KEY"}),
    ("Set SECRET_KEY = 'mysupersecret123!!'", {"API_KEY"}),
    ("password: P@ssw0rd_secure99", {"API_KEY"}),
    ("ACCESS_TOKEN = 'eyJhbGciOiJIUzI1NiJ9.payload.signature'", {"API_KEY"}),
    ("passwd: qwerty12345abc", {"API_KEY"}),

    # CREDIT_CARD
    ("Payment card: 4111 1111 1111 1111 expires 12/28.", {"CREDIT_CARD"}),
    ("Charge card 5500-0000-0000-0004.", {"CREDIT_CARD"}),
    ("Visa ending in 4242424242424242.", {"CREDIT_CARD"}),
    ("MC: 5105105105105100", {"CREDIT_CARD"}),
    ("Amex 3714 496353 98431 declined.", {"CREDIT_CARD"}),

    # MIXED PII
    ("User alice@domain.com (SSN: 111-22-3333) called 800-555-0100.", {"EMAIL", "SSN", "PHONE"}),
    ("Login failed for bob@corp.net from 192.168.0.5 using api_key='ABCDEFGHIJKLMN'", {"EMAIL", "IP_ADDRESS", "API_KEY"}),
    ("Record: SSN 999-88-7777, email charlie@test.com, card 4111111111111111.", {"SSN", "EMAIL", "CREDIT_CARD"}),

    # ── TRUE NEGATIVES (no PII — should NOT be detected) ──────────────────
    ("The weather in London is cloudy with a chance of rain.", set()),
    ("LoRA adapters use low-rank decomposition for efficient fine-tuning.", set()),
    ("The model achieved a perplexity of 14.02 on the validation set.", set()),
    ("AES-256-GCM provides authenticated encryption with 128-bit tags.", set()),
    ("HKDF derives keys deterministically from input key material.", set()),
    ("Please review the quarterly report for Q3 performance.", set()),
    ("The hash function SHA-256 produces a 32-byte digest.", set()),
    ("PyTorch version 2.0 introduced torch.compile for performance.", set()),
    ("This document contains no sensitive personal information.", set()),
    ("Deep learning models are trained using stochastic gradient descent.", set()),
    ("The RSA signature scheme uses probabilistic padding (PSS).", set()),
    ("Federated learning distributes training across multiple clients.", set()),
    ("Edge deployment requires hardware-efficient model compression.", set()),
    ("GDPR compliance requires data minimization and purpose limitation.", set()),
    ("Training loss converged to 2.64 after three epochs.", set()),
]

# Derive the evaluated PII entity classes from the ground-truth corpus
EVAL_PII_CLASSES: List[str] = sorted(
    list({pii_type for _, expected in GROUND_TRUTH_CORPUS for pii_type in expected})
)

# Reference alias to production ENTITIES_PATTERNS regexes for dataset loaders
_PII_PATTERNS: Dict[str, re.Pattern] = {
    k: v[0] for k, v in ENTITIES_PATTERNS.items() if k in EVAL_PII_CLASSES
}


# --------------------------------------------------------------------------
# Metric Calculation
# --------------------------------------------------------------------------

class PIIEvalResult:
    """Accumulates TP, FP, FN, TN for each evaluated PII class and computes metrics."""

    def __init__(self, pii_classes: List[str]):
        self.pii_classes = pii_classes
        self.tp: Dict[str, int] = {k: 0 for k in pii_classes}
        self.fp: Dict[str, int] = {k: 0 for k in pii_classes}
        self.fn: Dict[str, int] = {k: 0 for k in pii_classes}
        self.tn: Dict[str, int] = {k: 0 for k in pii_classes}

    def add(self, pii_type: str, detected: bool, expected: bool):
        if pii_type not in self.tp:
            return
        if detected and expected:
            self.tp[pii_type] += 1
        elif detected and not expected:
            self.fp[pii_type] += 1
        elif not detected and expected:
            self.fn[pii_type] += 1
        else:
            self.tn[pii_type] += 1

    def precision(self, pii_type: str) -> float:
        denom = self.tp[pii_type] + self.fp[pii_type]
        return self.tp[pii_type] / denom if denom else 0.0

    def recall(self, pii_type: str) -> float:
        denom = self.tp[pii_type] + self.fn[pii_type]
        return self.tp[pii_type] / denom if denom else 0.0

    def f1(self, pii_type: str) -> float:
        p = self.precision(pii_type)
        r = self.recall(pii_type)
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def accuracy(self, pii_type: str) -> float:
        total = self.tp[pii_type] + self.fp[pii_type] + self.fn[pii_type] + self.tn[pii_type]
        return (self.tp[pii_type] + self.tn[pii_type]) / total if total else 0.0

    def micro_avg(self) -> Dict[str, float]:
        total_tp = sum(self.tp.values())
        total_fp = sum(self.fp.values())
        total_fn = sum(self.fn.values())
        p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0
        r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        return {"precision": p, "recall": r, "f1": f1}

    def macro_avg(self) -> Dict[str, float]:
        n = len(self.pii_classes)
        p = sum(self.precision(t) for t in self.pii_classes) / n if n else 0.0
        r = sum(self.recall(t) for t in self.pii_classes) / n if n else 0.0
        f1 = sum(self.f1(t) for t in self.pii_classes) / n if n else 0.0
        return {"precision": p, "recall": r, "f1": f1}


def evaluate_pii_detection(verbose: bool = True) -> Dict[str, Any]:
    """
    Evaluates the production HybridPIIEngine against the ground-truth corpus.
    
    Returns structured results containing per-class metrics (precision, recall,
    F1, TP, FP, FN, TN), micro/macro averages, and explicit provenance metadata.
    """
    result = PIIEvalResult(EVAL_PII_CLASSES)
    sample_predictions = []

    for text, expected_types in GROUND_TRUTH_CORPUS:
        # Predictions come ONLY from the production PII engine
        detected = detect_pii(text)
        masked_text, mask_counts = mask_pii(text)

        for pii_type in EVAL_PII_CLASSES:
            det = pii_type in detected
            exp = pii_type in expected_types
            result.add(pii_type, det, exp)

        sample_predictions.append({
            "text_snippet": text[:80] + ("..." if len(text) > 80 else ""),
            "expected_pii": sorted(list(expected_types)),
            "detected_pii": sorted(list(detected.keys())),
            "masked_output": masked_text[:120] + ("..." if len(masked_text) > 120 else ""),
            "correct": set(detected.keys()) == expected_types,
        })

    # Build per-class metrics preserving precision, recall, F1, TP, FP, FN, TN
    per_class = {}
    for pii_type in EVAL_PII_CLASSES:
        per_class[pii_type] = {
            "precision": round(result.precision(pii_type), 4),
            "recall": round(result.recall(pii_type), 4),
            "f1": round(result.f1(pii_type), 4),
            "f1_score": round(result.f1(pii_type), 4),
            "accuracy": round(result.accuracy(pii_type), 4),
            "tp": result.tp[pii_type],
            "fp": result.fp[pii_type],
            "fn": result.fn[pii_type],
            "tn": result.tn[pii_type],
        }

    micro = result.micro_avg()
    macro = result.macro_avg()

    correct_samples = sum(1 for p in sample_predictions if p["correct"])
    sample_accuracy = correct_samples / len(GROUND_TRUTH_CORPUS)

    if verbose:
        print("\n" + "=" * 65)
        print("  Production PII/PHI Detection Evaluation Results (HybridPIIEngine)")
        print("=" * 65)
        print(f"\n{'PII Type':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5} {'TN':>5}")
        print("-" * 65)
        for pii_type, m in per_class.items():
            print(
                f"{pii_type:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1']:>10.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5} {m['tn']:>5}"
            )
        print("-" * 65)
        print(f"{'Micro Avg':<14} {micro['precision']:>10.4f} {micro['recall']:>10.4f} {micro['f1']:>10.4f}")
        print(f"{'Macro Avg':<14} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}")
        print(f"\nSample-level accuracy: {sample_accuracy:.2%} ({correct_samples}/{len(GROUND_TRUTH_CORPUS)} samples correct)")

    return {
        "detector_provenance": "Predictions were produced by the production HybridPIIEngine (src.security.pii_engine.HybridPIIEngine via detect_pii_advanced and mask_pii_advanced)",
        "metadata": {
            "evaluation_version": "1.1.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_size": len(GROUND_TRUTH_CORPUS),
            "pii_types_evaluated": EVAL_PII_CLASSES,
            "ground_truth_construction": "Manually labeled text corpus containing known synthetic PII snippets and non-PII control samples.",
            "evaluation_methodology": "Evaluation compares predictions from the production HybridPIIEngine against labeled ground-truth expected classes.",
            "production_detector_entry_point": "src.security.pii_engine.detect_pii_advanced / mask_pii_advanced (HybridPIIEngine)",
            "detector_provenance": "Predictions were produced by the production HybridPIIEngine (src.security.pii_engine.HybridPIIEngine)",
            "scope_disclaimer": "Evaluates input preprocessing PII/PHI detection and masking performance against a labeled ground-truth corpus. Does NOT measure live LLM memorization or PII generation leakage.",
            "compliance_scope": ["GDPR", "CCPA", "HIPAA"],
        },
        "per_class_metrics": per_class,
        "micro_average": {k: round(v, 4) for k, v in micro.items()},
        "macro_average": {k: round(v, 4) for k, v in macro.items()},
        "sample_accuracy": round(sample_accuracy, 4),
        "total_samples": len(GROUND_TRUTH_CORPUS),
        "correct_samples": correct_samples,
        "sample_predictions": sample_predictions,
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SecureLoRA PII Detection Metrics")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/pii_metrics.json",
        help="Path to save the JSON metrics report",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    metrics = evaluate_pii_detection(verbose=not args.quiet)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  PII metrics report saved -> {out_path}")
    return metrics


if __name__ == "__main__":
    main()
