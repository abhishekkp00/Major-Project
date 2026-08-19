"""
crypto_benchmark.py
===================
Real-world cryptographic performance benchmarks for the Secure Device-Bound
LoRA Fine-Tuning Framework.

Measures genuine wall-clock timing for:
  - AES-256-GCM encryption/decryption (streaming + block) at multiple sizes
  - HKDF key derivation latency
  - RSA-PSS sign / verify latency (2048-bit)
  - SHA-256 hashing throughput
  - Hardware fingerprint collection
  - Multi-pass file shredding

All timings are real — produced by time.perf_counter() over N_RUNS repetitions.
No values are hardcoded.

Usage:
    python -m src.evaluation.crypto_benchmark
    python -m src.evaluation.crypto_benchmark --output outputs/benchmarks/crypto_benchmark.json
"""

import os
import sys
import json
import time
import tempfile
import hashlib
import argparse
import statistics
import platform
import io
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when run directly
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from src.security.fingerprint import get_fingerprint_hash
from src.security.key_derivation import derive_key
from src.security.crypto import encrypt_stream, decrypt_stream, generate_key
from src.security.shred import shred_file

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_RUNS = 10          # repetitions per measurement
PAYLOAD_SIZES_KB = [16, 64, 256, 1024, 4096]   # KB — real adapter sizes
RSA_KEY_BITS = 2048
AES_KEY_BYTES = 32


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_key() -> bytes:
    return generate_key()


def _make_rsa_key():
    return rsa.generate_private_key(public_exponent=65537, key_size=RSA_KEY_BITS)


def _repeat(fn, n: int) -> List[float]:
    """Run fn() n times and return list of elapsed seconds."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return times


def _stats(samples: List[float]) -> Dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "stdev_ms": statistics.stdev(samples) * 1000 if len(samples) > 1 else 0.0,
        "min_ms": min(samples) * 1000,
        "max_ms": max(samples) * 1000,
        "n_runs": len(samples),
    }


# ---------------------------------------------------------------------------
# Individual benchmarks
# ---------------------------------------------------------------------------

def bench_hkdf_key_derivation() -> Dict[str, Any]:
    """Measure HKDF key derivation latency using the real derive_key function."""
    fp_hash = "a" * 64     # synthetic deterministic fingerprint hex
    salt = "benchmark_salt_v1"

    samples = _repeat(lambda: derive_key(fp_hash, salt), N_RUNS)
    return {
        "operation": "HKDF Key Derivation (SHA-256, 32-byte output)",
        "algorithm": "HKDF-SHA256",
        **_stats(samples),
    }


def bench_hardware_fingerprint() -> Dict[str, Any]:
    """Measure real hardware fingerprint collection + SHA-256 hashing."""
    samples = _repeat(get_fingerprint_hash, N_RUNS)
    return {
        "operation": "Hardware Fingerprint Collection + SHA-256",
        "sources": ["machine-id", "cpu-model", "disk-uuid"],
        **_stats(samples),
    }


def bench_aes_gcm_encryption(payload_kb: int) -> Dict[str, Any]:
    """Measure streaming AES-256-GCM encryption for a given payload size."""
    key = _make_key()
    payload = os.urandom(payload_kb * 1024)

    def _encrypt():
        instream = io.BytesIO(payload)
        outstream = io.BytesIO()
        encrypt_stream(instream, outstream, key)
        return outstream.getvalue()

    samples = _repeat(_encrypt, N_RUNS)
    throughput_mb_s = (payload_kb / 1024) / statistics.mean(samples)

    return {
        "operation": f"AES-256-GCM Streaming Encryption ({payload_kb} KB)",
        "algorithm": "AES-256-GCM",
        "payload_kb": payload_kb,
        "throughput_mb_per_s": round(throughput_mb_s, 3),
        **_stats(samples),
    }


def bench_aes_gcm_decryption(payload_kb: int) -> Dict[str, Any]:
    """Measure streaming AES-256-GCM decryption for a given payload size."""
    key = _make_key()
    payload = os.urandom(payload_kb * 1024)

    # Pre-encrypt once
    instream = io.BytesIO(payload)
    enc_buf = io.BytesIO()
    encrypt_stream(instream, enc_buf, key)
    ciphertext = enc_buf.getvalue()

    def _decrypt():
        instream2 = io.BytesIO(ciphertext)
        outstream = io.BytesIO()
        decrypt_stream(instream2, outstream, key)

    samples = _repeat(_decrypt, N_RUNS)
    throughput_mb_s = (payload_kb / 1024) / statistics.mean(samples)

    return {
        "operation": f"AES-256-GCM Streaming Decryption ({payload_kb} KB)",
        "algorithm": "AES-256-GCM",
        "payload_kb": payload_kb,
        "throughput_mb_per_s": round(throughput_mb_s, 3),
        **_stats(samples),
    }


def bench_aes_gcm_block(payload_kb: int) -> Dict[str, Any]:
    """Measure block-level (single-shot) AES-256-GCM for adapter-weight-sized payloads."""
    key = _make_key()
    aesgcm = AESGCM(key)
    payload = os.urandom(payload_kb * 1024)
    nonce = os.urandom(12)

    def _enc():
        return aesgcm.encrypt(nonce, payload, None)

    samples_enc = _repeat(_enc, N_RUNS)
    ciphertext = aesgcm.encrypt(nonce, payload, None)

    def _dec():
        return aesgcm.decrypt(nonce, ciphertext, None)

    samples_dec = _repeat(_dec, N_RUNS)

    return {
        "operation": f"AES-256-GCM Block Encrypt+Decrypt ({payload_kb} KB)",
        "algorithm": "AES-256-GCM",
        "payload_kb": payload_kb,
        "encrypt_mean_ms": round(statistics.mean(samples_enc) * 1000, 4),
        "decrypt_mean_ms": round(statistics.mean(samples_dec) * 1000, 4),
        "encrypt_stdev_ms": round(statistics.stdev(samples_enc) * 1000, 4) if len(samples_enc) > 1 else 0.0,
        "decrypt_stdev_ms": round(statistics.stdev(samples_dec) * 1000, 4) if len(samples_dec) > 1 else 0.0,
        "n_runs": N_RUNS,
    }


def bench_sha256_hash() -> Dict[str, Any]:
    """Measure SHA-256 hashing throughput across different file sizes."""
    results = []
    for size_kb in [64, 256, 1024, 4096]:
        data = os.urandom(size_kb * 1024)

        def _hash(d=data):
            hashlib.sha256(d).hexdigest()

        samples = _repeat(_hash, N_RUNS)
        throughput = (size_kb / 1024) / statistics.mean(samples)
        results.append({
            "payload_kb": size_kb,
            "mean_ms": round(statistics.mean(samples) * 1000, 4),
            "stdev_ms": round(statistics.stdev(samples) * 1000, 4) if len(samples) > 1 else 0.0,
            "throughput_mb_per_s": round(throughput, 3),
        })
    return {
        "operation": "SHA-256 File Integrity Hashing",
        "algorithm": "SHA-256",
        "sizes": results,
    }


def bench_rsa_pss_sign() -> Dict[str, Any]:
    """Measure RSA-PSS signing latency (real key generation + signing)."""
    private_key = _make_rsa_key()
    digest_hex = hashlib.sha256(os.urandom(256)).hexdigest()
    message = digest_hex.encode("utf-8")

    def _sign():
        private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

    samples = _repeat(_sign, N_RUNS)
    return {
        "operation": f"RSA-PSS Sign ({RSA_KEY_BITS}-bit)",
        "algorithm": f"RSA-{RSA_KEY_BITS}-PSS-SHA256",
        **_stats(samples),
    }


def bench_rsa_pss_verify() -> Dict[str, Any]:
    """Measure RSA-PSS verify latency (real signature verification)."""
    private_key = _make_rsa_key()
    public_key = private_key.public_key()
    digest_hex = hashlib.sha256(os.urandom(256)).hexdigest()
    message = digest_hex.encode("utf-8")
    signature = private_key.sign(
        message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
        hashes.SHA256(),
    )

    def _verify():
        public_key.verify(
            signature, message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

    samples = _repeat(_verify, N_RUNS)
    return {
        "operation": f"RSA-PSS Verify ({RSA_KEY_BITS}-bit)",
        "algorithm": f"RSA-{RSA_KEY_BITS}-PSS-SHA256",
        **_stats(samples),
    }


def bench_rsa_key_generation() -> Dict[str, Any]:
    """Measure RSA keypair generation time (one-time cost at packaging)."""
    samples = _repeat(lambda: rsa.generate_private_key(public_exponent=65537, key_size=RSA_KEY_BITS), N_RUNS)
    return {
        "operation": f"RSA Keypair Generation ({RSA_KEY_BITS}-bit)",
        "algorithm": f"RSA-{RSA_KEY_BITS}",
        **_stats(samples),
    }


def bench_file_shredding() -> Dict[str, Any]:
    """Measure multi-pass secure file shredding latency at multiple file sizes."""
    results = []
    for size_kb in [16, 64, 256]:
        data = os.urandom(size_kb * 1024)
        tmp_files = []

        # Create N_RUNS temp files with identical content
        for _ in range(N_RUNS):
            fd, path = tempfile.mkstemp(prefix="seclorabench_")
            os.close(fd)
            Path(path).write_bytes(data)
            tmp_files.append(Path(path))

        samples = []
        for tmp_path in tmp_files:
            t0 = time.perf_counter()
            shred_file(tmp_path, passes=3)
            samples.append(time.perf_counter() - t0)

        results.append({
            "payload_kb": size_kb,
            "passes": 3,
            "mean_ms": round(statistics.mean(samples) * 1000, 3),
            "stdev_ms": round(statistics.stdev(samples) * 1000, 3) if len(samples) > 1 else 0.0,
        })
    return {
        "operation": "3-Pass Secure File Shredding (DoD 5220.22-M style)",
        "algorithm": "urandom-overwrite × 3 + rename + unlink",
        "sizes": results,
    }


# ---------------------------------------------------------------------------
# End-to-end pipeline overhead benchmark
# ---------------------------------------------------------------------------

def bench_end_to_end_overhead() -> Dict[str, Any]:
    """
    Simulates a real full-pipeline run: key derivation → encrypt → hash → sign → verify → decrypt.
    Measures the total added security overhead vs. a plain file-copy baseline.
    """
    payload_kb = 512
    payload = os.urandom(payload_kb * 1024)

    # --- Baseline: plain file write/read ---
    def _baseline():
        fd, path = tempfile.mkstemp()
        os.close(fd)
        p = Path(path)
        try:
            p.write_bytes(payload)
            _ = p.read_bytes()
        finally:
            p.unlink(missing_ok=True)

    baseline_samples = _repeat(_baseline, N_RUNS)

    # --- Secure pipeline: derive key → encrypt → hash → sign → verify → decrypt ---
    private_key = _make_rsa_key()
    public_key = private_key.public_key()

    def _secure_pipeline():
        # 1. Key derivation
        fp = "a" * 64
        key = derive_key(fp, "bench_salt")

        # 2. Encrypt (streaming)
        enc_buf = io.BytesIO()
        encrypt_stream(io.BytesIO(payload), enc_buf, key)
        ciphertext = enc_buf.getvalue()

        # 3. Hash
        digest = hashlib.sha256(ciphertext).hexdigest()

        # 4. Sign
        sig = private_key.sign(
            digest.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # 5. Verify signature
        public_key.verify(
            sig, digest.encode(),
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
            hashes.SHA256(),
        )

        # 6. Decrypt
        decrypt_stream(io.BytesIO(ciphertext), io.BytesIO(), key)

    secure_samples = _repeat(_secure_pipeline, N_RUNS)

    overhead_ms = (statistics.mean(secure_samples) - statistics.mean(baseline_samples)) * 1000
    overhead_factor = statistics.mean(secure_samples) / max(statistics.mean(baseline_samples), 1e-9)

    return {
        "operation": "End-to-End Pipeline Security Overhead",
        "payload_kb": payload_kb,
        "baseline_mean_ms": round(statistics.mean(baseline_samples) * 1000, 3),
        "secure_pipeline_mean_ms": round(statistics.mean(secure_samples) * 1000, 3),
        "security_overhead_ms": round(overhead_ms, 3),
        "overhead_factor_x": round(overhead_factor, 2),
        "n_runs": N_RUNS,
    }


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def collect_system_info() -> Dict[str, Any]:
    import psutil
    return {
        "python_version": platform.python_version(),
        "os": platform.system(),
        "os_release": platform.release(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "cryptography_library": "cryptography (PyCA)",
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_benchmarks(verbose: bool = True) -> Dict[str, Any]:
    sections = {}

    def _log(msg: str):
        if verbose:
            print(f"  ⏱  {msg}")

    print("\n" + "=" * 60)
    print("  SecureLoRA Cryptographic Performance Benchmarks")
    print("=" * 60)

    print("\n[1/8] Hardware fingerprint ...")
    sections["hardware_fingerprint"] = bench_hardware_fingerprint()
    _log(f"mean={sections['hardware_fingerprint']['mean_ms']:.3f} ms")

    print("[2/8] HKDF key derivation ...")
    sections["hkdf_key_derivation"] = bench_hkdf_key_derivation()
    _log(f"mean={sections['hkdf_key_derivation']['mean_ms']:.3f} ms")

    print("[3/8] AES-256-GCM streaming encryption ...")
    sections["aes_gcm_encryption"] = [bench_aes_gcm_encryption(kb) for kb in PAYLOAD_SIZES_KB]
    for r in sections["aes_gcm_encryption"]:
        _log(f"{r['payload_kb']:5d} KB → mean={r['mean_ms']:.3f} ms  ({r['throughput_mb_per_s']} MB/s)")

    print("[4/8] AES-256-GCM streaming decryption ...")
    sections["aes_gcm_decryption"] = [bench_aes_gcm_decryption(kb) for kb in PAYLOAD_SIZES_KB]
    for r in sections["aes_gcm_decryption"]:
        _log(f"{r['payload_kb']:5d} KB → mean={r['mean_ms']:.3f} ms  ({r['throughput_mb_per_s']} MB/s)")

    print("[5/8] AES-256-GCM block (adapter-weight sizes) ...")
    sections["aes_gcm_block"] = [bench_aes_gcm_block(kb) for kb in [64, 256, 1024]]
    for r in sections["aes_gcm_block"]:
        _log(f"{r['payload_kb']:5d} KB → enc={r['encrypt_mean_ms']:.4f} ms  dec={r['decrypt_mean_ms']:.4f} ms")

    print("[6/8] SHA-256 file integrity hashing ...")
    sections["sha256_hashing"] = bench_sha256_hash()

    print("[7/8] RSA-PSS sign / verify / keygen ...")
    sections["rsa_pss_sign"] = bench_rsa_pss_sign()
    sections["rsa_pss_verify"] = bench_rsa_pss_verify()
    sections["rsa_key_generation"] = bench_rsa_key_generation()
    _log(f"sign mean={sections['rsa_pss_sign']['mean_ms']:.3f} ms")
    _log(f"verify mean={sections['rsa_pss_verify']['mean_ms']:.3f} ms")

    print("[8/8] End-to-end pipeline overhead vs. plain copy ...")
    sections["e2e_overhead"] = bench_end_to_end_overhead()
    _log(f"overhead={sections['e2e_overhead']['security_overhead_ms']:.2f} ms  ({sections['e2e_overhead']['overhead_factor_x']}x)")

    print("[+] File shredding latency ...")
    sections["file_shredding"] = bench_file_shredding()

    return {
        "metadata": {
            "benchmark_version": "1.0.0",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "n_runs_per_test": N_RUNS,
            "rsa_key_bits": RSA_KEY_BITS,
            "payload_sizes_kb": PAYLOAD_SIZES_KB,
            "system": collect_system_info(),
        },
        "results": sections,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Cryptographic Benchmarks")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmarks/crypto_benchmark.json",
        help="Path to save the JSON benchmark report",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    report = run_all_benchmarks(verbose=not args.quiet)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n  Benchmark report saved → {out_path}")
    return report


if __name__ == "__main__":
    main()
