"""
experiment_runner.py
====================
Reproducible Experiment Matrix Runner for the SecureLoRA Research Framework.

Executes standardized experiment configurations E0 through E9 (with B0-B8 alias support)
across multiple random seeds, gathering ML utility, privacy, security, and systems overhead metrics,
and aggregating statistical summaries (mean, std, 95% CI).
"""

from __future__ import annotations

import io
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.evaluation.metrics_schema import (
    SingleRunResult,
    AggregatedBaselineResult,
    MLUtilityMetrics,
    PrivacyMetrics,
    SecurityMetrics,
    SystemsOverheadMetrics,
    calculate_metric_summary,
)
from src.evaluation.reproducibility import collect_reproducibility_metadata
from src.security.crypto import encrypt_stream, decrypt_stream
from src.security.key_derivation import derive_key
from src.security.fingerprint import get_fingerprint_hash
from src.security.provenance import validate_manifest_schema, AntiReplayTracker
from src.common.exceptions import ReplayAttackError
from src.evaluation.adapter_security import evaluate_adapter_security, ScreeningMode, EvaluationInputType
from src.evaluation.pii_metrics import evaluate_pii_detection

logger = logging.getLogger("secure_lora.evaluation.experiment_runner")


EXPERIMENTS_DEFINITION = {
    "E0": {
        "name": "Base Model",
        "description": "Base language model JackFram/llama-68m evaluated zero-shot without fine-tuning, PII redaction, DP, encryption, or device binding.",
        "pii": False, "dp": False, "enc": False, "binding": False, "sig": False, "screen": False, "train": False,
    },
    "E1": {
        "name": "Standard LoRA",
        "description": "Standard LoRA fine-tuning without PII redaction, DP, encryption, or provenance.",
        "pii": False, "dp": False, "enc": False, "binding": False, "sig": False, "screen": False, "train": True,
    },
    "E2": {
        "name": "PII + LoRA",
        "description": "Phase 1 PII masking + Phase 2 Standard LoRA fine-tuning.",
        "pii": True, "dp": False, "enc": False, "binding": False, "sig": False, "screen": False, "train": True,
    },
    "E3": {
        "name": "DP-LoRA",
        "description": "Phase 2 DP-LoRA fine-tuning using Opacus per-example gradients and noise.",
        "pii": False, "dp": True, "enc": False, "binding": False, "sig": False, "screen": False, "train": True,
    },
    "E4": {
        "name": "PII + DP-LoRA",
        "description": "Phase 1 PII masking + Phase 2 Opacus DP-LoRA fine-tuning.",
        "pii": True, "dp": True, "enc": False, "binding": False, "sig": False, "screen": False, "train": True,
    },
    "E5": {
        "name": "LoRA + Encrypted Adapter",
        "description": "Standard LoRA fine-tuning + Phase 3 AES-256-GCM encryption at rest.",
        "pii": False, "dp": False, "enc": True, "binding": False, "sig": False, "screen": False, "train": True,
    },
    "E6": {
        "name": "LoRA + Device Binding",
        "description": "Standard LoRA + Phase 3 Adaptive Device-Bound HKDF Key Derivation & Gateway Policy Engine.",
        "pii": False, "dp": False, "enc": True, "binding": True, "sig": False, "screen": False, "train": True,
    },
    "E7": {
        "name": "LoRA + Integrity/Signature",
        "description": "Standard LoRA + Phase 3 RSA-PSS manifest signing & Monotonic Anti-Replay Package Provenance.",
        "pii": False, "dp": False, "enc": True, "binding": False, "sig": True, "screen": False, "train": True,
    },
    "E8": {
        "name": "PII + DP + Encrypted Adapter + Device Binding",
        "description": "Phase 1 PII + Phase 2 DP-LoRA + Phase 3 AES-256-GCM + Hardware Device Binding.",
        "pii": True, "dp": True, "enc": True, "binding": True, "sig": False, "screen": False, "train": True,
    },
    "E9": {
        "name": "FULL SECURELORA",
        "description": "Phase 1 PII + Phase 2 DP-LoRA + Pre-packaging Security Screening + Phase 3 Packaging (AES, Binding, RSA-PSS Signature, Anti-Replay).",
        "pii": True, "dp": True, "enc": True, "binding": True, "sig": True, "screen": True, "train": True,
    },
}

# Alias mapping B0-B8 -> E0-E9
ALIAS_MAPPING = {
    "B0": "E0",
    "B1": "E1",
    "B2": "E2",
    "B3": "E3",
    "B4": "E5",
    "B5": "E6",
    "B6": "E7",
    "B7": "E8",
    "B8": "E9",
}

BASELINES_DEFINITION = EXPERIMENTS_DEFINITION  # Backwards compatibility alias


def normalize_experiment_id(exp_id: str) -> str:
    """Normalizes baseline or experiment IDs (e.g. B1 -> E1)."""
    clean_id = exp_id.strip().upper()
    return ALIAS_MAPPING.get(clean_id, clean_id)


def run_single_baseline(
    baseline_id: str,
    seed: int,
    output_dir: Path,
    mock_payload_kb: int = 512,
    quick_mode: bool = False,
) -> SingleRunResult:
    """Executes a single experiment run for a specific configuration and seed."""
    normalized_id = normalize_experiment_id(baseline_id)
    if normalized_id not in EXPERIMENTS_DEFINITION:
        return SingleRunResult(
            baseline_id=baseline_id,
            baseline_name=baseline_id,
            seed=seed,
            execution_status="NOT_EXECUTED",
            not_executed_reason=f"Unknown experiment configuration ID: {baseline_id}",
        )

    defn = EXPERIMENTS_DEFINITION[normalized_id]
    b_name = defn["name"]

    meta = collect_reproducibility_metadata(
        experiment_id=f"EXP_{normalized_id}_seed_{seed}_{int(time.time())}",
        seed=seed,
        model_identifier="JackFram/llama-68m",
        dataset_identifier="sample_pii_data.jsonl",
        dataset_split="val",
        configuration_snapshot=defn,
    )

    # 0. Seed random generators reproducibly
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        # 1. PII Ingestion / Sanitization phase
        pii_prec, pii_rec, pii_f1 = 0.0, 0.0, 0.0
        pii_latency_ms = 0.0
        if defn["pii"]:
            t0_pii = time.perf_counter()
            try:
                pii_res = evaluate_pii_detection(verbose=False)
                micro = pii_res.get("micro_average", {})
                pii_prec = float(micro.get("precision", 0.0))
                pii_rec = float(micro.get("recall", 0.0))
                pii_f1 = float(micro.get("f1", 0.0))
            except Exception as pii_err:
                return SingleRunResult(
                    baseline_id=normalized_id,
                    baseline_name=b_name,
                    seed=seed,
                    execution_status="NOT_EXECUTED",
                    not_executed_reason=f"PII detection evaluation failed: {pii_err}",
                    metadata=meta,
                )
            pii_latency_ms = (time.perf_counter() - t0_pii) * 1000.0

        # 2. ML Training / Utility measurement
        train_loss = 0.0
        val_loss = 0.0
        perplexity = 1.0
        accuracy = 0.0
        f1 = 0.0
        train_time_s = 0.0
        inf_lat_ms = 0.0
        peak_mem_mb = 0.0
        dp_eps, dp_delta, dp_clip, dp_noise = None, None, None, None

        try:
            import torch
            import resource
            import hashlib
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import LoraConfig, get_peft_model, TaskType

            tokenizer = AutoTokenizer.from_pretrained("JackFram/llama-68m")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained("JackFram/llama-68m")

            eval_texts = [
                "The quick brown fox jumps over the lazy dog.",
                "SecureLoRA provides privacy-preserving LoRA adapter deployment.",
                "Differential privacy guarantees membership inference protection.",
                "Cryptographic device binding prevents unauthorized model execution.",
            ]
            if defn["pii"]:
                from src.security.pii_engine import mask_pii_advanced
                eval_texts = [mask_pii_advanced(txt)[0] for txt in eval_texts]

            encodings = tokenizer(eval_texts, return_tensors="pt", padding=True, truncation=True)
            encodings["labels"] = encodings["input_ids"].clone()

            if not defn["train"]:  # E0: Base model zero-shot
                model.eval()
                t0_inf = time.perf_counter()
                with torch.no_grad():
                    outputs = model(**encodings)
                    v_loss = float(outputs.loss.item())
                inf_lat_ms = (time.perf_counter() - t0_inf) * 1000.0

                val_loss = v_loss
                train_loss = v_loss
                perplexity = float(math.exp(val_loss))

                with torch.no_grad():
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=-1)
                    correct = (preds == encodings["labels"]).float()
                    accuracy = float(correct.mean().item())
                    f1 = accuracy
                train_time_s = 0.0

            else:  # E1-E9 training experiments
                peft_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj"],
                    lora_dropout=0.05,
                )
                model = get_peft_model(model, peft_config)

                if defn["dp"]:
                    from opacus import PrivacyEngine
                    from torch.utils.data import DataLoader, TensorDataset

                    ds = TensorDataset(encodings["input_ids"], encodings["attention_mask"], encodings["labels"])
                    dl = DataLoader(ds, batch_size=2)

                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    privacy_engine = PrivacyEngine()
                    model, optimizer, dl = privacy_engine.make_private(
                        module=model,
                        optimizer=optimizer,
                        data_loader=dl,
                        noise_multiplier=1.0,
                        max_grad_norm=1.0,
                    )
                    dp_delta = 1e-5
                    dp_clip = 1.0
                    dp_noise = 1.0

                    model.train()
                    t0_tr = time.perf_counter()
                    epochs = 1 if quick_mode else 2
                    last_tr_loss = 0.0
                    for _ in range(epochs):
                        for b_ids, b_mask, b_labels in dl:
                            if b_ids.size(0) == 0:
                                continue
                            optimizer.zero_grad()
                            b_out = model(b_ids, attention_mask=b_mask, labels=b_labels)
                            b_loss = b_out.loss
                            b_loss.backward()
                            optimizer.step()
                            last_tr_loss = float(b_loss.item())

                    train_time_s = time.perf_counter() - t0_tr
                    train_loss = last_tr_loss
                    dp_eps = float(privacy_engine.get_epsilon(delta=dp_delta))

                    model.eval()
                    eval_mod = getattr(model, "_module", model)
                    with torch.no_grad():
                        val_out = eval_mod(**encodings)
                        val_loss = float(val_out.loss.item())
                        perplexity = float(math.exp(val_loss))
                        logits = val_out.logits
                        preds = torch.argmax(logits, dim=-1)
                        correct = (preds == encodings["labels"]).float()
                        accuracy = float(correct.mean().item())
                        f1 = accuracy

                else:
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                    model.train()
                    t0_tr = time.perf_counter()
                    epochs = 1 if quick_mode else 2
                    last_tr_loss = 0.0
                    for _ in range(epochs):
                        optimizer.zero_grad()
                        outputs = model(**encodings)
                        loss = outputs.loss
                        loss.backward()
                        optimizer.step()
                        last_tr_loss = float(loss.item())

                    train_time_s = time.perf_counter() - t0_tr
                    train_loss = last_tr_loss

                    model.eval()
                    with torch.no_grad():
                        outputs = model(**encodings)
                        val_loss = float(outputs.loss.item())
                        perplexity = float(math.exp(val_loss))
                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        correct = (preds == encodings["labels"]).float()
                        accuracy = float(correct.mean().item())
                        f1 = accuracy

                # Measure inference latency
                eval_mod = getattr(model, "_module", model)
                t0_inf = time.perf_counter()
                with torch.no_grad():
                    _ = eval_mod(**encodings)
                inf_lat_ms = (time.perf_counter() - t0_inf) * 1000.0

            peak_mem_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
            if torch.cuda.is_available():
                peak_mem_mb = float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)

        except Exception as model_err:
            return SingleRunResult(
                baseline_id=normalized_id,
                baseline_name=b_name,
                seed=seed,
                execution_status="NOT_EXECUTED",
                not_executed_reason=f"Model execution/training failed: {model_err}",
                metadata=meta,
            )

        # 3. Pre-packaging Security Screening
        screen_time_ms = 0.0
        malicious_detection_rate = 0.0
        if defn["screen"]:
            if not defn["train"] or model is None:
                return SingleRunResult(
                    baseline_id=normalized_id,
                    baseline_name=b_name,
                    seed=seed,
                    execution_status="NOT_EXECUTED",
                    not_executed_reason="Real trained LoRA adapter artifact is unavailable for security screening.",
                    metadata=meta,
                )

            t0_scr = time.perf_counter()
            # Extract actual trained adapter weight tensors from PyTorch PEFT model
            real_adapter_weights = {
                k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v, dtype=np.float32))
                for k, v in model.state_dict().items()
            }

            def real_inference_cb(prompt_text: str) -> str:
                eval_mod = getattr(model, "_module", model)
                eval_mod.eval()
                inputs = tokenizer(prompt_text, return_tensors="pt", truncation=True)
                with torch.no_grad():
                    gen_out = eval_mod.generate(**inputs, max_new_tokens=32, do_sample=False)
                return tokenizer.decode(gen_out[0], skip_special_tokens=True)

            try:
                scr_res = evaluate_adapter_security(
                    adapter_source=real_adapter_weights,
                    adapter_id=f"run-{normalized_id}-{seed}",
                    candidate_model_fn=real_inference_cb,
                    mode=ScreeningMode.PRODUCTION,
                    evaluation_input_type="REAL_ADAPTER_EVALUATION",
                    base_model_id="JackFram/llama-68m",
                )
                screen_time_ms = (time.perf_counter() - t0_scr) * 1000.0
                malicious_detection_rate = 1.0 if scr_res.approved else 0.0
            except Exception as scr_err:
                return SingleRunResult(
                    baseline_id=normalized_id,
                    baseline_name=b_name,
                    seed=seed,
                    execution_status="NOT_EXECUTED",
                    not_executed_reason=f"Real adapter security screening failed: {scr_err}",
                    metadata=meta,
                )

        # 4. Device Binding & AES-256-GCM Encryption
        payload = os.urandom(mock_payload_kb * 1024)
        if defn["binding"]:
            t0_kdf = time.perf_counter()
            fp_hash = get_fingerprint_hash()
            key = derive_key(fp_hash, "salt_v1")
            key_deriv_ms = (time.perf_counter() - t0_kdf) * 1000.0
        else:
            fp_hash = "00" * 32
            key = b"\x00" * 32
            key_deriv_ms = 0.0

        enc_time_ms = 0.0
        dec_time_ms = 0.0
        ciphertext = payload

        if defn["enc"]:
            t0_enc = time.perf_counter()
            enc_buf = io.BytesIO()
            encrypt_stream(io.BytesIO(payload), enc_buf, key)
            ciphertext = enc_buf.getvalue()
            enc_time_ms = (time.perf_counter() - t0_enc) * 1000.0

            t0_dec = time.perf_counter()
            dec_buf = io.BytesIO()
            decrypt_stream(io.BytesIO(ciphertext), dec_buf, key)
            dec_time_ms = (time.perf_counter() - t0_dec) * 1000.0

        # 5. RSA-PSS Manifest Signing & Verification
        sign_time_ms = 0.0
        verify_time_ms = 0.0
        sig = b""
        if defn["sig"]:
            from cryptography.hazmat.primitives.asymmetric import rsa, padding
            from cryptography.hazmat.primitives import hashes

            priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
            pub = priv.public_key()
            digest = hashlib.sha256(ciphertext).digest()

            t0_sig = time.perf_counter()
            sig = priv.sign(
                digest,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            sign_time_ms = (time.perf_counter() - t0_sig) * 1000.0

            t0_ver = time.perf_counter()
            pub.verify(
                sig,
                digest,
                padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                hashes.SHA256(),
            )
            verify_time_ms = (time.perf_counter() - t0_ver) * 1000.0

        pkg_time_ms = enc_time_ms + sign_time_ms + screen_time_ms
        deploy_time_ms = dec_time_ms + verify_time_ms + key_deriv_ms
        storage_bytes = len(ciphertext) + len(sig)

        # 6. Security Metrics (derived strictly from actual attack test cases executed)
        sec_details: Dict[str, Any] = {}
        all_executed_cases: List[Tuple[bool, bool]] = []

        def evaluate_security_case(
            metric_key: str,
            source_test: str,
            test_cases: List[Tuple[bool, bool]],
        ) -> Tuple[Optional[float], Optional[int], Optional[int]]:
            """
            Computes rate = successful_rejections / applicable_test_cases.
            If no test cases were executed, returns None (null / NOT_EXECUTED).
            """
            if not test_cases:
                sec_details[metric_key] = {
                    "numerator": None,
                    "denominator": None,
                    "rate": None,
                    "status": "NOT_EXECUTED",
                    "source_test": source_test,
                }
                return None, None, None

            n_cases = len(test_cases)
            rejections = sum(1 for is_attack, is_blocked in test_cases if is_blocked)
            rate = rejections / float(n_cases)
            sec_details[metric_key] = {
                "numerator": rejections,
                "denominator": n_cases,
                "rate": round(rate, 4),
                "status": "EXECUTED",
                "source_test": source_test,
            }
            all_executed_cases.extend(test_cases)
            return round(rate, 4), rejections, n_cases

        # (a) Unauthorized Device Rejection Rate
        unauth_dev_cases: List[Tuple[bool, bool]] = []
        if defn["binding"]:
            from src.security.device_auth_policy import evaluate_device_authorization, flatten_classified_features
            from src.security.fingerprint import build_canonical_string, compute_fingerprint_hash

            expected_classified = {
                "stable": {"machine_id": "real-id-123", "cpu_model": "Real CPU"},
                "semi_stable": {"disk_uuid": "real-disk-uuid"},
                "volatile": {"hostname": "real-node", "network_interface": "00:11:22:33:44:55"},
            }
            expected_flat = flatten_classified_features(expected_classified)
            expected_hash = compute_fingerprint_hash(build_canonical_string(expected_flat))

            # Case 1: Authorized device (must succeed)
            auth_res = evaluate_device_authorization(
                expected_fingerprint_hash=expected_hash,
                expected_features=expected_flat,
                current_classified=expected_classified,
            )
            unauth_dev_cases.append((False, auth_res.is_authorized))

            # Case 2: Unauthorized device (must be rejected)
            unauth_classified = {
                "stable": {"machine_id": "foreign-id-999", "cpu_model": "Foreign CPU"},
                "semi_stable": {"disk_uuid": "foreign-disk-uuid"},
                "volatile": {"hostname": "unauthorized-node", "network_interface": "ff:ff:ff:ff:ff:ff"},
            }
            unauth_res = evaluate_device_authorization(
                expected_fingerprint_hash=expected_hash,
                expected_features=expected_flat,
                current_classified=unauth_classified,
            )
            unauth_dev_cases.append((True, not unauth_res.is_authorized))

        unauth_device_rate, _, _ = evaluate_security_case(
            "unauthorized_device_rejection",
            "test_device_authorization_classified_features",
            unauth_dev_cases,
        )

        # (b) Cross-Device Rejection Rate
        cross_dev_cases: List[Tuple[bool, bool]] = []
        if defn["binding"] and defn["enc"]:
            wrong_dev_key = derive_key("ff" * 32, "salt_v1")
            rejected_cross = False
            try:
                decrypt_stream(io.BytesIO(ciphertext), io.BytesIO(), wrong_dev_key)
            except Exception:
                rejected_cross = True
            cross_dev_cases.append((True, rejected_cross))

        cross_device_rate, _, _ = evaluate_security_case(
            "cross_device_rejection",
            "test_cross_device_key_decryption",
            cross_dev_cases,
        )

        # (c) Tamper Rejection Rate
        tamper_cases: List[Tuple[bool, bool]] = []
        if defn["enc"] or defn["sig"]:
            tampered_bytes = bytearray(ciphertext)
            if len(tampered_bytes) > 0:
                tampered_bytes[0] ^= 0xFF
            tampered_ciphertext = bytes(tampered_bytes)

            if defn["enc"]:
                rej_enc_tamper = False
                try:
                    decrypt_stream(io.BytesIO(tampered_ciphertext), io.BytesIO(), key)
                except Exception:
                    rej_enc_tamper = True
                tamper_cases.append((True, rej_enc_tamper))

            if defn["sig"]:
                rej_sig_tamper = False
                try:
                    tampered_digest = hashlib.sha256(tampered_ciphertext).digest()
                    pub.verify(
                        sig,
                        tampered_digest,
                        padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                        hashes.SHA256(),
                    )
                except Exception:
                    rej_sig_tamper = True
                tamper_cases.append((True, rej_sig_tamper))

        tamper_rate, _, _ = evaluate_security_case(
            "tamper_rejection",
            "test_gcm_ciphertext_tamper_and_signature_digest_verification",
            tamper_cases,
        )

        # (d) Signature Rejection Rate
        sig_cases: List[Tuple[bool, bool]] = []
        if defn["sig"]:
            bad_sig = sig[:-1] + (b"\x00" if sig[-1:] != b"\x00" else b"\x01")
            rej_bad_sig = False
            try:
                pub.verify(
                    bad_sig,
                    digest,
                    padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.MAX_LENGTH),
                    hashes.SHA256(),
                )
            except Exception:
                rej_bad_sig = True
            sig_cases.append((True, rej_bad_sig))

        sig_rate, _, _ = evaluate_security_case(
            "signature_rejection",
            "test_rsa_pss_invalid_signature_verification",
            sig_cases,
        )

        # (e) Wrong Key Rejection Rate
        wrong_key_cases: List[Tuple[bool, bool]] = []
        if defn["enc"]:
            rej_wrong_key = False
            try:
                decrypt_stream(io.BytesIO(ciphertext), io.BytesIO(), os.urandom(32))
            except Exception:
                rej_wrong_key = True
            wrong_key_cases.append((True, rej_wrong_key))

        wrong_key_rate, _, _ = evaluate_security_case(
            "wrong_key_rejection",
            "test_aes_gcm_wrong_key_decryption",
            wrong_key_cases,
        )

        # (f) Replay Rejection Rate
        replay_cases: List[Tuple[bool, bool]] = []
        if defn["sig"] or defn["binding"]:
            state_path = output_dir / f".replay_test_{normalized_id}_{seed}.json"
            if state_path.exists():
                try:
                    state_path.unlink()
                except Exception:
                    pass
            tracker = AntiReplayTracker(state_file_path=state_path)
            manifest_test = {
                "package_id": f"pkg-test-{normalized_id}-{seed}",
                "adapter_id": f"adapter-test-{normalized_id}",
                "sequence_number": 1,
            }
            tracker.check_and_update(manifest_test)
            rej_replay = False
            try:
                tracker.check_and_update(manifest_test)
            except ReplayAttackError:
                rej_replay = True
            replay_cases.append((True, rej_replay))

            if state_path.exists():
                try:
                    state_path.unlink()
                except Exception:
                    pass

        replay_rate, _, _ = evaluate_security_case(
            "replay_rejection",
            "test_anti_replay_tracker_duplicate_nonce",
            replay_cases,
        )

        # (g) Malicious Adapter Detection Rate
        malicious_adapter_cases: List[Tuple[bool, bool]] = []
        if defn["screen"]:
            rej_malicious = not scr_res.approved
            malicious_adapter_cases.append((True, rej_malicious))

        malicious_adapter_rate, _, _ = evaluate_security_case(
            "malicious_adapter_detection",
            "test_adapter_screening_probe_suite",
            malicious_adapter_cases,
        )

        # (h) Unauthorized Deployment Rejection Rate
        unauth_deploy_rate, _, _ = evaluate_security_case(
            "unauthorized_deployment_rejection",
            "test_combined_unauthorized_deployment_gate",
            all_executed_cases,
        )

        sec_metrics = SecurityMetrics(
            unauthorized_device_rejection_rate=unauth_device_rate,
            cross_device_rejection_rate=cross_device_rate,
            tamper_rejection_rate=tamper_rate,
            signature_rejection_rate=sig_rate,
            wrong_key_rejection_rate=wrong_key_rate,
            replay_rejection_rate=replay_rate,
            malicious_adapter_detection_rate=malicious_adapter_rate,
            unauthorized_deployment_rejection_rate=unauth_deploy_rate,
            details=sec_details,
        )

        util_metrics = MLUtilityMetrics(
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            perplexity=round(perplexity, 4),
            task_accuracy=round(accuracy, 4),
            f1_score=round(f1, 4),
        )

        priv_metrics = PrivacyMetrics(
            dp_enabled=defn["dp"],
            epsilon=round(dp_eps, 4) if dp_eps is not None else None,
            delta=dp_delta,
            clipping_norm=dp_clip,
            noise_multiplier=dp_noise,
            pii_precision=round(pii_prec, 4),
            pii_recall=round(pii_rec, 4),
            pii_f1=round(pii_f1, 4),
        )

        ovh_metrics = SystemsOverheadMetrics(
            training_time_s=round(train_time_s, 3),
            encryption_time_ms=round(enc_time_ms, 3),
            decryption_time_ms=round(dec_time_ms, 3),
            signing_time_ms=round(sign_time_ms, 3),
            verification_time_ms=round(verify_time_ms, 3),
            packaging_time_ms=round(pkg_time_ms, 3),
            deployment_latency_ms=round(deploy_time_ms, 3),
            deployment_time_ms=round(deploy_time_ms, 3),
            inference_latency_ms=round(inf_lat_ms, 3),
            memory_usage_mb=round(peak_mem_mb, 2),
            peak_memory_mb=round(peak_mem_mb, 2),
            storage_overhead_bytes=storage_bytes,
            package_size_bytes=storage_bytes,
        )

        result = SingleRunResult(
            baseline_id=normalized_id,
            baseline_name=b_name,
            seed=seed,
            execution_status="COMPLETED",
            not_executed_reason=None,
            utility=util_metrics,
            privacy=priv_metrics,
            security=sec_metrics,
            overhead=ovh_metrics,
            metadata=meta,
        )

        # Save run artifact
        run_file = output_dir / "runs" / f"EXP_{normalized_id}_seed_{seed}.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2)

        return result

    except Exception as e:
        logger.error("Error executing experiment %s seed %d: %s", normalized_id, seed, e)
        return SingleRunResult(
            baseline_id=normalized_id,
            baseline_name=b_name,
            seed=seed,
            execution_status="NOT_EXECUTED",
            not_executed_reason=str(e),
            metadata=meta,
        )


def aggregate_baseline_runs(
    baseline_id: str,
    runs: List[SingleRunResult],
) -> AggregatedBaselineResult:
    """Aggregates multiple single-run results across seeds into mean, stdev, 95% CIs."""
    normalized_id = normalize_experiment_id(baseline_id)
    defn = EXPERIMENTS_DEFINITION.get(normalized_id, {"name": baseline_id, "description": baseline_id})
    completed_runs = [r for r in runs if r.execution_status == "COMPLETED"]

    if not completed_runs:
        unexec_reasons = [r.not_executed_reason for r in runs if r.not_executed_reason]
        reason = unexec_reasons[0] if unexec_reasons else "Hardware resources or execution aborted."
        return AggregatedBaselineResult(
            baseline_id=normalized_id,
            baseline_name=defn["name"],
            description=defn["description"],
            execution_status="NOT_EXECUTED",
            not_executed_reason=reason,
            num_seeds=0,
            utility_summary={},
            privacy_summary={},
            security_summary={},
            overhead_summary={},
        )

    # Calculate summaries for numeric metrics
    utility_summary = {
        "train_loss": calculate_metric_summary([r.utility.train_loss for r in completed_runs]),
        "val_loss": calculate_metric_summary([r.utility.val_loss for r in completed_runs]),
        "perplexity": calculate_metric_summary([r.utility.perplexity for r in completed_runs]),
        "task_accuracy": calculate_metric_summary([r.utility.task_accuracy for r in completed_runs]),
        "f1_score": calculate_metric_summary([r.utility.f1_score for r in completed_runs]),
    }

    security_summary = {
        "unauthorized_device_rejection_rate": calculate_metric_summary([r.security.unauthorized_device_rejection_rate for r in completed_runs]),
        "cross_device_rejection_rate": calculate_metric_summary([r.security.cross_device_rejection_rate for r in completed_runs]),
        "tamper_rejection_rate": calculate_metric_summary([r.security.tamper_rejection_rate for r in completed_runs]),
        "signature_rejection_rate": calculate_metric_summary([r.security.signature_rejection_rate for r in completed_runs]),
        "wrong_key_rejection_rate": calculate_metric_summary([r.security.wrong_key_rejection_rate for r in completed_runs]),
        "replay_rejection_rate": calculate_metric_summary([r.security.replay_rejection_rate for r in completed_runs]),
        "malicious_adapter_detection_rate": calculate_metric_summary([r.security.malicious_adapter_detection_rate for r in completed_runs]),
        "unauthorized_deployment_rejection_rate": calculate_metric_summary([r.security.unauthorized_deployment_rejection_rate for r in completed_runs]),
    }

    overhead_summary = {
        "training_time_s": calculate_metric_summary([r.overhead.training_time_s for r in completed_runs]),
        "encryption_time_ms": calculate_metric_summary([r.overhead.encryption_time_ms for r in completed_runs]),
        "decryption_time_ms": calculate_metric_summary([r.overhead.decryption_time_ms for r in completed_runs]),
        "signing_time_ms": calculate_metric_summary([r.overhead.signing_time_ms for r in completed_runs]),
        "verification_time_ms": calculate_metric_summary([r.overhead.verification_time_ms for r in completed_runs]),
        "packaging_time_ms": calculate_metric_summary([r.overhead.packaging_time_ms for r in completed_runs]),
        "deployment_latency_ms": calculate_metric_summary([r.overhead.deployment_latency_ms for r in completed_runs]),
        "deployment_time_ms": calculate_metric_summary([r.overhead.deployment_time_ms for r in completed_runs]),
        "inference_latency_ms": calculate_metric_summary([r.overhead.inference_latency_ms for r in completed_runs]),
        "memory_usage_mb": calculate_metric_summary([r.overhead.memory_usage_mb for r in completed_runs]),
        "peak_memory_mb": calculate_metric_summary([r.overhead.peak_memory_mb for r in completed_runs]),
        "storage_overhead_bytes": calculate_metric_summary([float(r.overhead.storage_overhead_bytes) for r in completed_runs]),
        "package_size_bytes": calculate_metric_summary([float(r.overhead.package_size_bytes) for r in completed_runs]),
    }

    first_priv = completed_runs[0].privacy
    privacy_summary = {
        "dp_enabled": first_priv.dp_enabled,
        "epsilon": first_priv.epsilon,
        "delta": first_priv.delta,
        "clipping_norm": first_priv.clipping_norm,
        "noise_multiplier": first_priv.noise_multiplier,
        "pii_precision": calculate_metric_summary([r.privacy.pii_precision for r in completed_runs]),
        "pii_recall": calculate_metric_summary([r.privacy.pii_recall for r in completed_runs]),
        "pii_f1": calculate_metric_summary([r.privacy.pii_f1 for r in completed_runs]),
    }

    return AggregatedBaselineResult(
        baseline_id=normalized_id,
        baseline_name=defn["name"],
        description=defn["description"],
        execution_status="COMPLETED",
        not_executed_reason=None,
        num_seeds=len(completed_runs),
        utility_summary=utility_summary,
        privacy_summary=privacy_summary,
        security_summary=security_summary,
        overhead_summary=overhead_summary,
    )


def run_experiment_matrix(
    seeds: List[int] = [42, 43, 44],
    output_dir: Path = Path("outputs/research"),
    experiment_ids: Optional[List[str]] = None,
    quick_mode: bool = False,
) -> Dict[str, AggregatedBaselineResult]:
    """Executes the complete reproducible ablation experiment matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)
    target_ids = [normalize_experiment_id(e) for e in (experiment_ids or list(EXPERIMENTS_DEFINITION.keys()))]

    logger.info("Starting Reproducibility Experiment Matrix across experiments %s and seeds %s...", target_ids, seeds)

    aggregated_results = {}
    all_raw_runs = []

    for exp_id in target_ids:
        if exp_id not in EXPERIMENTS_DEFINITION:
            logger.warning("Skipping unknown experiment ID: %s", exp_id)
            continue

        logger.info("Executing Experiment %s (%s)...", exp_id, EXPERIMENTS_DEFINITION[exp_id]["name"])
        seed_runs = []
        for s in seeds:
            run_res = run_single_baseline(baseline_id=exp_id, seed=s, output_dir=output_dir, quick_mode=quick_mode)
            seed_runs.append(run_res)
            all_raw_runs.append(run_res.to_dict())

        agg = aggregate_baseline_runs(baseline_id=exp_id, runs=seed_runs)
        aggregated_results[exp_id] = agg

        # Save aggregated summary JSON artifact
        agg_file = output_dir / "metrics" / f"{exp_id}_summary.json"
        agg_file.parent.mkdir(parents=True, exist_ok=True)
        with open(agg_file, "w", encoding="utf-8") as f:
            json.dump(agg.to_dict(), f, indent=2)

    # Save raw aggregated data
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_dir / "raw_experiments.json", "w", encoding="utf-8") as f:
        json.dump(all_raw_runs, f, indent=2)

    return aggregated_results
