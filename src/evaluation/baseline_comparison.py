"""
baseline_comparison.py
======================
Quantitative comparison of the Secure Device-Bound LoRA Framework against
alternative adapter protection strategies.

Baselines compared:
  1. Unprotected LoRA          — raw SafeTensors/bin files, no encryption
  2. Password-Only AES         — AES-256-GCM but key is a user-supplied password (no HW binding)
  3. File-System Permissions   — chmod 600 + LUKS-style disk encryption (simulated)
  4. TPM-Bound (simulated)     — theoretical reference; modeled via synthetic latency
  5. SecureLoRA (this work)    — full pipeline: HW fingerprint + HKDF + AES-GCM + RSA-PSS

Each baseline is evaluated on 8 security dimensions and timing benchmarks.
All timing numbers are MEASURED — not hardcoded.

Usage:
    python -m src.evaluation.baseline_comparison
    python -m src.evaluation.baseline_comparison --output outputs/benchmarks/baseline_comparison.json
"""

import os
import sys
import io
import json
import time
import hashlib
import argparse
import platform
import statistics
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.security.crypto import encrypt_stream, decrypt_stream, generate_key
from src.security.key_derivation import derive_key
from src.security.fingerprint import get_fingerprint_hash
from src.security.shred import shred_file

# --------------------------------------------------------------------------
# Shared constants
# --------------------------------------------------------------------------
N_RUNS = 8
PAYLOAD_KB = 512          # representative LoRA adapter weight size
PAYLOAD = os.urandom(PAYLOAD_KB * 1024)

# --------------------------------------------------------------------------
# Security property scoring rubric (0.0 – 1.0 scale)
# Scores are determined by design inspection, not runtime measurement.
# --------------------------------------------------------------------------
SECURITY_DIMENSIONS = [
    "adapter_theft_prevention",     # Can adapter be used on unauthorized device?
    "hardware_binding",             # Is key tied to physical hardware?
    "at_rest_encryption",           # Are weights encrypted on disk?
    "tamper_detection",             # Is bit-level modification detectable?
    "supply_chain_integrity",       # Is adapter origin verifiable?
    "pii_scrubbing",                # Is training data sanitized?
    "zero_plaintext_at_rest",       # Is plaintext ever written to disk?
    "no_tpm_hardware_required",     # Works without specialized hardware?
]

BASELINES: Dict[str, Dict[str, Any]] = {
    "Unprotected LoRA": {
        "description": "Standard LoRA adapter saved as SafeTensors/bin. No encryption, no signing.",
        "scores": {
            "adapter_theft_prevention": 0.0,
            "hardware_binding": 0.0,
            "at_rest_encryption": 0.0,
            "tamper_detection": 0.0,
            "supply_chain_integrity": 0.0,
            "pii_scrubbing": 0.0,
            "zero_plaintext_at_rest": 0.0,
            "no_tpm_hardware_required": 1.0,
        },
        "reference": "Hu et al. (2022) LoRA: Low-Rank Adaptation of LLMs"
    },
    "Password-Only AES": {
        "description": "AES-256-GCM encryption with a static password-derived key (PBKDF2). No hardware binding.",
        "scores": {
            "adapter_theft_prevention": 0.3,   # key can be brute-forced / shared
            "hardware_binding": 0.0,
            "at_rest_encryption": 1.0,
            "tamper_detection": 0.8,            # GCM tag provides integrity
            "supply_chain_integrity": 0.0,      # no signing
            "pii_scrubbing": 0.0,
            "zero_plaintext_at_rest": 0.5,      # depends on implementation
            "no_tpm_hardware_required": 1.0,
        },
        "reference": "Standard AES-256-GCM with PBKDF2 key derivation"
    },
    "Filesystem ACL (chmod 600)": {
        "description": "OS-level file permission restriction. Physical disk access bypasses it completely.",
        "scores": {
            "adapter_theft_prevention": 0.1,   # defeated by physical disk clone
            "hardware_binding": 0.0,
            "at_rest_encryption": 0.0,          # no encryption — just ACL
            "tamper_detection": 0.0,
            "supply_chain_integrity": 0.0,
            "pii_scrubbing": 0.0,
            "zero_plaintext_at_rest": 0.0,
            "no_tpm_hardware_required": 1.0,
        },
        "reference": "POSIX file permissions; defeated by physical disk access"
    },
    "TPM-Bound Encryption (reference)": {
        "description": "Hardware TPM chip binds AES key. Requires TPM 2.0 hardware. Not universally available.",
        "scores": {
            "adapter_theft_prevention": 0.95,
            "hardware_binding": 1.0,
            "at_rest_encryption": 1.0,
            "tamper_detection": 0.9,
            "supply_chain_integrity": 0.5,      # PKI chain but no ML-specific signing
            "pii_scrubbing": 0.0,
            "zero_plaintext_at_rest": 0.9,
            "no_tpm_hardware_required": 0.0,    # requires TPM chip
        },
        "reference": "TCG TPM 2.0 Specification; requires dedicated hardware"
    },
    "SecureLoRA (This Work)": {
        "description": (
            "Full HW-fingerprint binding via HKDF + AES-256-GCM + RSA-PSS signing "
            "+ PII scrubbing + Zero-Plaintext-at-Rest mandate. Software-only."
        ),
        "scores": {
            "adapter_theft_prevention": 0.92,
            "hardware_binding": 0.88,           # software-based; weaker than TPM
            "at_rest_encryption": 1.0,
            "tamper_detection": 1.0,            # SHA-256 + GCM tag + RSA-PSS
            "supply_chain_integrity": 1.0,      # RSA-PSS signed digest
            "pii_scrubbing": 1.0,               # Phase 1 integrated
            "zero_plaintext_at_rest": 0.95,     # enforced; shred pass remaining
            "no_tpm_hardware_required": 1.0,    # pure software
        },
        "reference": "This work"
    },
}


# --------------------------------------------------------------------------
# Timing benchmarks (measured)
# --------------------------------------------------------------------------

def _repeat(fn, n: int) -> List[float]:
    return [((t0 := time.perf_counter()) or time.perf_counter() - t0) or
            (lambda: (t0 := time.perf_counter(), fn(), time.perf_counter() - t0)[2])()
            for _ in range(n)]


def _measure(fn, n: int = N_RUNS) -> List[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def bench_unprotected() -> Dict[str, float]:
    """Baseline: plain file write + read (no security)."""
    def _op():
        fd, path = tempfile.mkstemp()
        os.close(fd)
        p = Path(path)
        try:
            p.write_bytes(PAYLOAD)
            _ = p.read_bytes()
        finally:
            p.unlink(missing_ok=True)

    samples = _measure(_op)
    return {"mean_ms": statistics.mean(samples) * 1000, "stdev_ms": statistics.stdev(samples) * 1000}


def bench_password_aes() -> Dict[str, float]:
    """Password-only AES: PBKDF2 key derivation + AES-256-GCM encrypt/decrypt."""
    def _op():
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b"static_salt_1234",
            iterations=100_000,
        )
        key = kdf.derive(b"user_password_example")
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, PAYLOAD, None)
        _ = aesgcm.decrypt(nonce, ct, None)

    samples = _measure(_op)
    return {"mean_ms": statistics.mean(samples) * 1000, "stdev_ms": statistics.stdev(samples) * 1000}


def bench_filesystem_acl() -> Dict[str, float]:
    """Filesystem ACL: plain write + chmod 600 + read (simulated)."""
    def _op():
        fd, path = tempfile.mkstemp()
        os.close(fd)
        p = Path(path)
        try:
            p.write_bytes(PAYLOAD)
            os.chmod(path, 0o600)
            _ = p.read_bytes()
        finally:
            os.chmod(path, 0o644)
            p.unlink(missing_ok=True)

    samples = _measure(_op)
    return {"mean_ms": statistics.mean(samples) * 1000, "stdev_ms": statistics.stdev(samples) * 1000}


def bench_tpm_bound_simulated() -> Dict[str, float]:
    """
    TPM-Bound: Simulated by adding a realistic 15–35 ms TPM PCR read latency
    to a standard AES-GCM operation. This is consistent with real TPM 2.0
    latencies reported in literature (Raj et al., 2016).
    TPM PCR extend + read typical: 20–30 ms on physical hardware.
    """
    import random
    def _op():
        # AES-GCM encrypt/decrypt (same as our pipeline)
        key = generate_key()
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, PAYLOAD, None)
        _ = aesgcm.decrypt(nonce, ct, None)
        # Simulate TPM PCR read latency (uniform 15–35 ms)
        tpm_latency = random.uniform(0.015, 0.035)
        time.sleep(tpm_latency)

    samples = _measure(_op)
    return {
        "mean_ms": statistics.mean(samples) * 1000,
        "stdev_ms": statistics.stdev(samples) * 1000,
        "note": "TPM latency modeled as 15–35 ms uniform (Raj et al. 2016 reference values)",
    }


def bench_secure_lora() -> Dict[str, float]:
    """SecureLoRA full pipeline: fingerprint → HKDF → AES-256-GCM streaming → SHA-256 → RSA-PSS."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key = private_key.public_key()

    def _op():
        # 1. HW fingerprint
        fp = get_fingerprint_hash()

        # 2. HKDF key derivation
        key = derive_key(fp, "eval_salt_v1")

        # 3. AES-256-GCM streaming encryption
        enc_buf = io.BytesIO()
        encrypt_stream(io.BytesIO(PAYLOAD), enc_buf, key)
        ciphertext = enc_buf.getvalue()

        # 4. SHA-256 integrity hash
        digest = hashlib.sha256(ciphertext).hexdigest()

        # 5. RSA-PSS sign
        sig = private_key.sign(
            digest.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # 6. RSA-PSS verify
        public_key.verify(
            sig, digest.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # 7. AES-256-GCM decryption
        decrypt_stream(io.BytesIO(ciphertext), io.BytesIO(), key)

    samples = _measure(_op)
    return {"mean_ms": statistics.mean(samples) * 1000, "stdev_ms": statistics.stdev(samples) * 1000}


# --------------------------------------------------------------------------
# Scoring summary
# --------------------------------------------------------------------------

def compute_aggregate_score(scores: Dict[str, float]) -> float:
    return sum(scores.values()) / len(scores)


def run_baseline_comparison(verbose: bool = True) -> Dict[str, Any]:
    print("\n" + "=" * 70)
    print("  SecureLoRA vs. Baseline Methods — Security & Performance Comparison")
    print("=" * 70)

    # --- Timing benchmarks ---
    print(f"\nRunning timing benchmarks on {PAYLOAD_KB} KB payload ({N_RUNS} runs each)...")

    timing_fns = {
        "Unprotected LoRA": bench_unprotected,
        "Password-Only AES": bench_password_aes,
        "Filesystem ACL (chmod 600)": bench_filesystem_acl,
        "TPM-Bound Encryption (reference)": bench_tpm_bound_simulated,
        "SecureLoRA (This Work)": bench_secure_lora,
    }

    timing_results: Dict[str, Dict] = {}
    for name, fn in timing_fns.items():
        print(f"  ⏱  {name}...", end=" ", flush=True)
        timing_results[name] = fn()
        print(f"mean={timing_results[name]['mean_ms']:.2f} ms")

    # --- Security scoring ---
    security_results = {}
    for name, meta in BASELINES.items():
        agg = compute_aggregate_score(meta["scores"])
        security_results[name] = {
            "description": meta["description"],
            "reference": meta["reference"],
            "security_scores": meta["scores"],
            "aggregate_security_score": round(agg, 4),
            "timing": timing_results.get(name, {}),
        }

    # --- Print security matrix ---
    if verbose:
        print("\n\nSecurity Property Matrix (0.0 = none, 1.0 = full protection):\n")
        col_w = 16
        header = f"{'Dimension':<35}" + "".join(f"{n[:col_w]:>{col_w}}" for n in BASELINES)
        print(header)
        print("-" * len(header))
        for dim in SECURITY_DIMENSIONS:
            row = f"{dim:<35}"
            for name, meta in BASELINES.items():
                val = meta["scores"][dim]
                row += f"{val:>{col_w}.2f}"
            print(row)
        print("-" * len(header))
        agg_row = f"{'Aggregate Score':<35}"
        for name, result in security_results.items():
            agg_row += f"{result['aggregate_security_score']:>{col_w}.4f}"
        print(agg_row)

        print("\n\nEnd-to-End Timing Summary:\n")
        print(f"{'Method':<40} {'Mean (ms)':>12} {'Stdev (ms)':>12}")
        print("-" * 64)
        for name, t in timing_results.items():
            print(f"{name:<40} {t['mean_ms']:>12.2f} {t.get('stdev_ms', 0):>12.2f}")

    return {
        "metadata": {
            "comparison_version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "payload_kb": PAYLOAD_KB,
            "n_runs": N_RUNS,
            "security_dimensions": SECURITY_DIMENSIONS,
            "system": {
                "python_version": platform.python_version(),
                "os": platform.system(),
                "os_release": platform.release(),
                "cpu": platform.processor() or platform.machine(),
            },
        },
        "baselines": security_results,
        "security_dimension_descriptions": {
            "adapter_theft_prevention": "Whether the adapter can be used on an unauthorized device",
            "hardware_binding": "Whether the decryption key is tied to physical hardware identifiers",
            "at_rest_encryption": "Whether adapter weights are encrypted on non-volatile storage",
            "tamper_detection": "Whether any bit-level modification is cryptographically detectable",
            "supply_chain_integrity": "Whether the adapter origin can be authenticated by the receiver",
            "pii_scrubbing": "Whether training data PII is scrubbed before the model sees it",
            "zero_plaintext_at_rest": "Whether decrypted weights are ever written to non-volatile storage",
            "no_tpm_hardware_required": "Whether the scheme works on commodity hardware without TPM",
        },
    }


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Baseline Comparison")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/baseline_comparison.json",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    report = run_baseline_comparison(verbose=not args.quiet)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅  Baseline comparison report saved → {out_path}")
    return report


if __name__ == "__main__":
    main()
