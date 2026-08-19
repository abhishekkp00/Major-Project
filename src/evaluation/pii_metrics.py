"""
pii_metrics.py
==============
Real PII/PHI detection precision, recall, and F1-score evaluation for the
Secure Device-Bound LoRA Fine-Tuning Framework.

Constructs a labeled ground-truth test corpus covering:
  - Social Security Numbers (SSN)
  - Email addresses
  - Phone numbers
  - Passwords / API keys / secrets
  - IP addresses
  - Credit card patterns
  - Named credential patterns

Runs the framework's actual preprocessing module against this corpus and
computes sklearn-style classification metrics (per-class and micro/macro averages).

Usage:
    python -m src.evaluation.pii_metrics
    python -m src.evaluation.pii_metrics --output outputs/benchmarks/pii_metrics.json
"""

import sys
import re
import json
import argparse
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# --------------------------------------------------------------------------
# The exact same regexes that the pipeline uses (read from preprocessing)
# --------------------------------------------------------------------------
# These mirror the real patterns in the framework so metrics are genuine
_PII_PATTERNS: Dict[str, re.Pattern] = {
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "EMAIL": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "PHONE": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b"
    ),
    "IP_ADDRESS": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    "API_KEY": re.compile(
        r"(?i)(?:api[_\-]?key|secret[_\-]?key|access[_\-]?token|password|passwd|pwd)"
        r"[\s:=]+[\'\"]?([A-Za-z0-9\-_/+.]{8,})"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4\d{3}|5[1-5]\d{2}|6(?:011|5\d{2})|3[47]\d{2})"
        r"[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"
    ),
}

MASK_MAP = {
    "SSN": "[MASKED_SSN]",
    "EMAIL": "[MASKED_EMAIL]",
    "PHONE": "[MASKED_PHONE]",
    "IP_ADDRESS": "[MASKED_IP]",
    "API_KEY": "[MASKED_SECRET]",
    "CREDIT_CARD": "[MASKED_CARD]",
}


from src.security.pii_engine import detect_pii_advanced, mask_pii_advanced

def detect_pii(text: str) -> Dict[str, List[str]]:
    """Returns dict of detected PII type → list of matched strings using the Advanced Hybrid PII Engine."""
    return detect_pii_advanced(text)


def mask_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Applies masking to all detected PII using the Advanced Hybrid PII Engine. Returns (masked_text, counts_per_type)."""
    return mask_pii_advanced(text)


# --------------------------------------------------------------------------
# Ground-truth test corpus
# --------------------------------------------------------------------------
# Each entry: (text, expected_pii_types_present: set)
# "expected" = set of PII type strings that SHOULD be detected in that text.

GROUND_TRUTH_CORPUS: List[Tuple[str, set]] = [
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
    ("CC: bob+filter@company.co.uk; charlie@test.io", {"EMAIL", "EMAIL"}),
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


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

class PIIEvalResult:
    def __init__(self):
        self.tp: Dict[str, int] = {k: 0 for k in _PII_PATTERNS}
        self.fp: Dict[str, int] = {k: 0 for k in _PII_PATTERNS}
        self.fn: Dict[str, int] = {k: 0 for k in _PII_PATTERNS}
        self.tn: Dict[str, int] = {k: 0 for k in _PII_PATTERNS}

    def add(self, pii_type: str, detected: bool, expected: bool):
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
        types = list(_PII_PATTERNS.keys())
        p = sum(self.precision(t) for t in types) / len(types)
        r = sum(self.recall(t) for t in types) / len(types)
        f1 = sum(self.f1(t) for t in types) / len(types)
        return {"precision": p, "recall": r, "f1": f1}


def evaluate_pii_detection(verbose: bool = True) -> Dict[str, Any]:
    """Runs evaluation against the ground truth corpus and returns metrics."""
    result = PIIEvalResult()
    sample_predictions = []

    for text, expected_types in GROUND_TRUTH_CORPUS:
        detected = detect_pii(text)
        masked_text, mask_counts = mask_pii(text)

        for pii_type in _PII_PATTERNS:
            det = pii_type in detected
            exp = pii_type in expected_types
            result.add(pii_type, det, exp)

        sample_predictions.append({
            "text_snippet": text[:80] + ("..." if len(text) > 80 else ""),
            "expected_pii": sorted(expected_types),
            "detected_pii": sorted(detected.keys()),
            "masked_output": masked_text[:120] + ("..." if len(masked_text) > 120 else ""),
            "correct": set(detected.keys()) == expected_types,
        })

    # Build per-class metrics
    per_class = {}
    for pii_type in _PII_PATTERNS:
        per_class[pii_type] = {
            "precision": round(result.precision(pii_type), 4),
            "recall": round(result.recall(pii_type), 4),
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
        print("\n" + "=" * 60)
        print("  PII/PHI Detection Evaluation Results")
        print("=" * 60)
        print(f"\n{'PII Type':<14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>5} {'FP':>5} {'FN':>5}")
        print("-" * 60)
        for pii_type, m in per_class.items():
            print(
                f"{pii_type:<14} {m['precision']:>10.4f} {m['recall']:>10.4f} "
                f"{m['f1_score']:>10.4f} {m['tp']:>5} {m['fp']:>5} {m['fn']:>5}"
            )
        print("-" * 60)
        print(f"{'Micro Avg':<14} {micro['precision']:>10.4f} {micro['recall']:>10.4f} {micro['f1']:>10.4f}")
        print(f"{'Macro Avg':<14} {macro['precision']:>10.4f} {macro['recall']:>10.4f} {macro['f1']:>10.4f}")
        print(f"\nSample-level accuracy: {sample_accuracy:.2%} ({correct_samples}/{len(GROUND_TRUTH_CORPUS)} samples correct)")

    return {
        "metadata": {
            "evaluation_version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "corpus_size": len(GROUND_TRUTH_CORPUS),
            "pii_types_evaluated": list(_PII_PATTERNS.keys()),
            "methodology": "Regex pattern matching against labeled ground-truth corpus",
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

    print(f"\n  PII metrics report saved → {out_path}")
    return metrics


if __name__ == "__main__":
    main()
