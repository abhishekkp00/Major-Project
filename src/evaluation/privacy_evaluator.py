"""
privacy_evaluator.py
====================
Reproducible Privacy & Generation-Level Memorization Evaluation Pipeline for SecureLoRA.

Evaluates two distinct privacy dimensions:
  1. Input PII Sanitization/Redaction:
     Measures pre-training input redaction (precision, recall, F1) across known PII spans.
  2. Generation-Level Memorization Leakage (Canary Extraction):
     Evaluates whether trained models memorize and emit sensitive canary secrets (SSNs, API keys,
     medical IDs, credentials) when probed with held-out prefix prompts that do NOT contain the secret.

Threat Model:
  Adversary queries the model with context prefixes (e.g., "Patient record for John Doe. SSN: ")
  and checks if the completion reproduces the secret canary that was present in training data.

Theoretical vs. Empirical Privacy Separation:
  Differential privacy parameters (epsilon, delta, clipping norm, noise multiplier) are reported
  separately as algorithmic bounds and are NEVER inferred from empirical extraction rates alone.

Canonical Outputs:
  outputs/evaluation/privacy/
    ├── base_model.json
    ├── lora.json
    ├── dp_lora.json
    ├── securelora.json
    ├── canary_benchmark.json
    └── comparison.json
"""

import os
import sys
import json
import argparse
import logging
import difflib
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_sources.dataset_registry import dataset_registry
from src.security.pii_engine import detect_pii_advanced
from src.orchestrator.model_registry import model_registry
from src.orchestrator.inference_service import generate_base, generate_securelora
from src.evaluation.canary_benchmark import (
    get_canary_records,
    score_extraction,
    normalize_secret,
    CANARY_RECORDS
)

logger = logging.getLogger("secure_lora.evaluation.privacy_evaluator")

PRIVACY_OUT_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "privacy"

# Global lazy cache for direct HuggingFace evaluation models
_EVAL_CACHE: Dict[str, Any] = {}


def _get_eval_model_and_tokenizer(model_name: str = "JackFram/llama-68m", adapter_path: Optional[str] = None):
    """Loads or retrieves cached model and tokenizer for evaluation."""
    cache_key = f"{model_name}:{adapter_path or 'none'}"
    if cache_key in _EVAL_CACHE:
        return _EVAL_CACHE[cache_key]

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=False)
        base_model.eval()

        if adapter_path and Path(adapter_path).exists():
            from peft import PeftModel
            model = PeftModel.from_pretrained(base_model, adapter_path)
            model.eval()
        else:
            model = base_model

        _EVAL_CACHE[cache_key] = (model, tokenizer)
        return model, tokenizer
    except Exception as exc:
        logger.warning("Could not load direct evaluation model %s: %s", model_name, exc)
        return None, None


def _generate_model_output(
    variant: str,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None,
    base_model_name: str = "JackFram/llama-68m",
    adapter_path: Optional[str] = None,
    allow_direct_loading: bool = False
) -> Tuple[str, bool]:
    """
    Generates text from model instance for variant:
      - 'base_model'
      - 'lora'
      - 'dp_lora'
      - 'securelora'
    Returns tuple (raw_output_text, success_boolean).
    """
    config = generation_config or {}
    max_new_tokens = int(config.get("max_new_tokens", 20))
    temperature = float(config.get("temperature", 0.0))
    seed = int(config.get("seed", 42))

    # 1. First attempt generation via active ModelRegistry if populated
    info = model_registry.get_info()
    has_registry_model = info.get("base_model") is not None or info.get("peft_model") is not None

    if has_registry_model:
        try:
            if variant == "base_model":
                out = generate_base(prompt, {"max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed})
                return out, True
            elif variant in ("lora", "securelora"):
                if info.get("peft_model") is not None:
                    out = generate_securelora(prompt, {"max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed})
                    return out, True
                return "", False
            elif variant == "dp_lora":
                dp_enabled = info.get("dp_enabled", False)
                adapter_type = str(info.get("adapter_type") or info.get("adapter_id") or "")
                if (dp_enabled or "dp" in adapter_type.lower()) and info.get("peft_model") is not None:
                    out = generate_securelora(prompt, {"max_new_tokens": max_new_tokens, "temperature": temperature, "seed": seed})
                    return out, True
                return "", False
        except Exception as exc:
            logger.debug("Registry generation failed for variant %s: %s", variant, exc)

    if not allow_direct_loading:
        return "", False

    # 2. Fall back to direct HuggingFace evaluation models where constructed and allowed
    try:
        import torch
        if variant == "base_model":
            model, tokenizer = _get_eval_model_and_tokenizer(base_model_name, None)
            if model is None or tokenizer is None:
                return "", False

            if hasattr(model, "disable_adapter"):
                with model.disable_adapter():
                    inputs = tokenizer(prompt, return_tensors="pt")
                    with torch.no_grad():
                        out_ids = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=temperature > 0.05,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
            else:
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=temperature > 0.05,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id
                    )
            full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            # Strip prompt prefix from output to isolate continuation
            continuation = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            return continuation, True

        elif variant == "lora":
            # Check provided adapter path, or fallback to default candidate paths
            target_adapter = adapter_path
            if not target_adapter or not Path(target_adapter).exists():
                candidates = [
                    Path("outputs/evaluation/privacy/canary_lora_adapter"),
                    Path("outputs/final_adapter"),
                    Path("outputs/notebook_adapter")
                ]
                for c in candidates:
                    if c.exists() and (c / "adapter_config.json").exists():
                        target_adapter = str(c)
                        break

            if not target_adapter or not Path(target_adapter).exists():
                return "", False

            model, tokenizer = _get_eval_model_and_tokenizer(base_model_name, target_adapter)
            if model is None or tokenizer is None:
                return "", False

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0.05,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            full_text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
            continuation = full_text[len(prompt):] if full_text.startswith(prompt) else full_text
            return continuation, True

        elif variant in ("dp_lora", "securelora"):
            # Strict rule: Unconstructed variants remain NOT_EXECUTED.
            # Do NOT simulate or fabricate generations.
            return "", False

    except Exception as exc:
        logger.warning("Direct generation for variant '%s' failed: %s", variant, exc)
        return "", False

    return "", False


def calculate_memorization_extraction_metrics(
    raw_outputs: List[str],
    canaries: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Computes generation-level memorization leakage metrics:
      - exact_extraction_rate: fraction of canaries emitted verbatim.
      - normalized_extraction_rate: fraction of canaries emitted under normalization.
      - exact_matches_count: total exact matches.
      - partial_matches_count: total partial / normalized matches.
      - total_canaries_tested: total canaries probed.
      - mean_normalized_similarity: average character-level sequence similarity.
      - extraction_by_secret_type: breakdown across secret categories.
    """
    total = len(canaries)
    if total == 0 or len(raw_outputs) == 0:
        return {
            "exact_extraction_rate": 0.0,
            "normalized_extraction_rate": 0.0,
            "exact_matches_count": 0,
            "partial_matches_count": 0,
            "total_canaries_tested": 0,
            "mean_normalized_similarity": 0.0,
            "extraction_by_secret_type": {}
        }

    exact_matches = 0
    norm_matches = 0
    similarities = []
    by_type: Dict[str, Dict[str, Any]] = {}

    for idx, canary in enumerate(canaries):
        out_text = raw_outputs[idx] if idx < len(raw_outputs) else ""
        secret = canary.get("ground_truth_secret", "")
        sec_type = canary.get("secret_type", "UNKNOWN")

        score = score_extraction(secret, out_text)

        if score["exact_match"]:
            exact_matches += 1
        if score["normalized_match"]:
            norm_matches += 1
        similarities.append(score["normalized_similarity"])

        if sec_type not in by_type:
            by_type[sec_type] = {"total": 0, "exact": 0, "normalized": 0}
        by_type[sec_type]["total"] += 1
        if score["exact_match"]:
            by_type[sec_type]["exact"] += 1
        if score["normalized_match"]:
            by_type[sec_type]["normalized"] += 1

    exact_rate = round(exact_matches / total, 4)
    norm_rate = round(norm_matches / total, 4)
    mean_sim = round(sum(similarities) / len(similarities), 4) if similarities else 0.0

    return {
        "exact_extraction_rate": exact_rate,
        "normalized_extraction_rate": norm_rate,
        "exact_matches_count": exact_matches,
        "partial_matches_count": norm_matches,
        "total_canaries_tested": total,
        "mean_normalized_similarity": mean_sim,
        "extraction_by_secret_type": by_type,
        "threat_model": {
            "attack_vector": "Black-box prefix probing completion",
            "prompt_protocol": "Held-out context prompt (secret string strictly excluded from prompt)",
            "decoding_parameters": "Greedy decoding (temperature=0.0, max_new_tokens=20)",
            "target": "Probes whether fine-tuning caused verbatim or normalized sensitive token emission",
            "limitations": "Prefix probing tests memorization under specific context prompts. Lack of extraction under one prompt does not guarantee zero extraction under adversarial search, greedy search vs sampling, or membership inference attacks."
        }
    }


def calculate_privacy_metrics(
    raw_outputs: List[str],
    ground_truth_records: Optional[List[Dict[str, Any]]] = None,
    ground_truth_available: bool = False
) -> Dict[str, Any]:
    """
    Calculates PII leakage rate, entity counts, types, PII-free response rate,
    and when ground_truth_available is True: precision, recall, F1, FPR, FNR.

    NOTE: Measures regex/classifier PII detection on output text (input/output sanitization),
    NOT held-out generation memorization. For memorization extraction, see calculate_memorization_extraction_metrics.
    """
    total_records = len(raw_outputs)
    if total_records == 0:
        return {
            "pii_leakage_rate": 0.0,
            "pii_free_response_rate": 1.0,
            "pii_entity_count": 0,
            "records_containing_pii": 0,
            "pii_entity_types": {}
        }

    records_with_pii = 0
    total_entities_detected = 0
    pii_entity_types: Dict[str, int] = {}

    tp_total = 0
    fp_total = 0
    fn_total = 0
    tn_total = 0

    for idx, raw_text in enumerate(raw_outputs):
        detected_dict = detect_pii_advanced(raw_text)

        detected_entities = []
        for ptype, vals in detected_dict.items():
            for val in vals:
                detected_entities.append((ptype.upper(), val.lower()))
                pii_entity_types[ptype] = pii_entity_types.get(ptype, 0) + 1

        if detected_entities:
            records_with_pii += 1
            total_entities_detected += len(detected_entities)

        if ground_truth_available and ground_truth_records and idx < len(ground_truth_records):
            gt_entities = ground_truth_records[idx].get("pii_entities", [])
            matched_gt = set()

            for gt_idx, gt in enumerate(gt_entities):
                gt_text = str(gt.get("text", "")).lower()
                match = False
                for det_t, det_v in detected_entities:
                    if det_v == gt_text or (gt_text and gt_text in det_v):
                        match = True
                        break

                if match:
                    tp_total += 1
                    matched_gt.add(gt_idx)
                else:
                    fn_total += 1

            fp = max(0, len(detected_entities) - len(matched_gt))
            fp_total += fp
            if not gt_entities and not detected_entities:
                tn_total += 1

    leakage_rate = round(records_with_pii / total_records, 4)
    pii_free_rate = round(1.0 - leakage_rate, 4)

    metrics = {
        "pii_leakage_rate": leakage_rate,
        "pii_free_response_rate": pii_free_rate,
        "pii_entity_count": total_entities_detected,
        "records_containing_pii": records_with_pii,
        "pii_entity_types": pii_entity_types
    }

    if ground_truth_available:
        precision = round(tp_total / (tp_total + fp_total), 4) if (tp_total + fp_total) > 0 else 1.0
        recall = round(tp_total / (tp_total + fn_total), 4) if (tp_total + fn_total) > 0 else 1.0
        f1 = round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0
        fpr = round(fp_total / max(1, fp_total + tn_total), 4)
        fnr = round(fn_total / max(1, tp_total + fn_total), 4)

        metrics.update({
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr,
            "true_positives": tp_total,
            "false_positives": fp_total,
            "false_negatives": fn_total
        })

    return metrics


def evaluate_privacy_pipeline(
    dataset_id: str = "canary_synthetic_benchmark",
    split: str = "test",
    samples: int = 30,
    seed: int = 42,
    base_model_name: str = "JackFram/llama-68m",
    adapter_id: str = "secure_lora_adapter",
    output_dir: Optional[Path] = None,
    allow_direct_loading: bool = False
) -> Dict[str, Any]:
    """
    Runs generation-level memorization extraction evaluation for all 4 variants
    (base_model, lora, dp_lora, securelora) and saves canonical output files.
    """
    out_dir = Path(output_dir) if output_dir else PRIVACY_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load controlled canary benchmark
    canaries = get_canary_records(count=samples)
    now_iso = datetime.now(timezone.utc).isoformat()

    # Save canonical canary benchmark artifact
    benchmark_file = out_dir / "canary_benchmark.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump({
            "benchmark_name": "SecureLoRA Controlled Canary Memorization Benchmark",
            "version": "1.0.0",
            "total_canaries": len(canaries),
            "threat_model": "Adversary probes model with held-out prefix context; secret strings are omitted from prompt.",
            "canaries": canaries
        }, f, indent=2)

    variants = ["base_model", "lora", "dp_lora", "securelora"]
    results_by_variant = {}

    gen_config = {
        "temperature": 0.0,
        "max_new_tokens": 20,
        "seed": seed
    }

    # Theoretical DP parameters for DP-LoRA and SecureLoRA specifications
    dp_specs = {
        "dp_enabled": True,
        "epsilon": 2.443,
        "delta": 1e-05,
        "clipping_norm": 1.0,
        "noise_multiplier": 1.2,
        "accounting_method": "Rényi DP (Opacus)",
        "formal_guarantee_note": "Differential privacy (ε, δ) is an algorithmic worst-case bound guaranteed by DP-SGD noise addition and gradient clipping. It is not inferred from, and must not be conflated with, empirical extraction rates."
    }

    for variant in variants:
        raw_outputs = []
        raw_probes = []
        executed = True

        for canary in canaries:
            prompt = canary["prompt_prefix"]
            secret = canary["ground_truth_secret"]

            text_out, ok = _generate_model_output(
                variant=variant,
                prompt=prompt,
                generation_config=gen_config,
                base_model_name=base_model_name,
                adapter_path=adapter_id if variant != "base_model" else None,
                allow_direct_loading=allow_direct_loading
            )

            if not ok:
                executed = False
                break

            raw_outputs.append(text_out)
            score = score_extraction(secret, text_out)
            raw_probes.append({
                "canary_id": canary["canary_id"],
                "domain": canary["domain"],
                "secret_type": canary["secret_type"],
                "prompt_prefix": prompt,
                "ground_truth_secret": secret,
                "generated_continuation": text_out,
                "exact_match": score["exact_match"],
                "normalized_match": score["normalized_match"],
                "normalized_similarity": score["normalized_similarity"],
                "match_type": score["match_type"]
            })

        variant_file = out_dir / f"{variant}.json"

        if executed and raw_outputs:
            mem_metrics = calculate_memorization_extraction_metrics(raw_outputs, canaries)
            pii_detect_metrics = calculate_privacy_metrics(raw_outputs)

            combined_metrics = dict(pii_detect_metrics)
            combined_metrics["generation_memorization_leakage"] = mem_metrics

            variant_data = {
                "dataset": "SecureLoRA Controlled Canary Memorization Benchmark",
                "dataset_id": "canary_synthetic_benchmark",
                "model": base_model_name,
                "adapter": adapter_id if variant != "base_model" else "none",
                "seed": seed,
                "sample_count": len(canaries),
                "timestamp": now_iso,
                "status": "EXECUTED",
                "ground_truth_available": True,
                "threat_model": mem_metrics["threat_model"],
                "configuration": {
                    "split": split,
                    "subset_size": len(canaries),
                    "seed": seed,
                    "temperature": gen_config["temperature"],
                    "max_new_tokens": gen_config["max_new_tokens"],
                    "decoding_method": "greedy"
                },
                "differential_privacy_guarantee": {
                    "dp_enabled": False,
                    "formal_guarantee_note": "No DP-SGD differential privacy applied to this baseline."
                },
                "observed_extraction_metrics": {
                    "exact_extraction_rate": mem_metrics["exact_extraction_rate"],
                    "normalized_extraction_rate": mem_metrics["normalized_extraction_rate"],
                    "exact_matches_count": mem_metrics["exact_matches_count"],
                    "partial_matches_count": mem_metrics["partial_matches_count"],
                    "total_canaries_tested": mem_metrics["total_canaries_tested"],
                    "mean_normalized_similarity": mem_metrics["mean_normalized_similarity"],
                    "extraction_by_secret_type": mem_metrics["extraction_by_secret_type"]
                },
                "metrics": combined_metrics,
                "raw_probes": raw_probes
            }
        else:
            # STRICT NO FABRICATION: Unconstructed variants remain strictly NOT_EXECUTED
            is_dp_variant = variant in ("dp_lora", "securelora")
            dp_info = dp_specs if is_dp_variant else {
                "dp_enabled": False,
                "formal_guarantee_note": "No DP mechanism configured."
            }

            variant_data = {
                "dataset": "SecureLoRA Controlled Canary Memorization Benchmark",
                "dataset_id": "canary_synthetic_benchmark",
                "model": base_model_name,
                "adapter": adapter_id if variant != "base_model" else "none",
                "seed": seed,
                "sample_count": len(canaries),
                "timestamp": now_iso,
                "status": "NOT_EXECUTED",
                "reason": f"Checkpoint or adapter instance for variant '{variant}' has not been trained or constructed on this canary benchmark.",
                "ground_truth_available": True,
                "configuration": {
                    "split": split,
                    "subset_size": len(canaries),
                    "seed": seed
                },
                "differential_privacy_guarantee": dp_info,
                "observed_extraction_metrics": None,
                "metrics": None,
                "raw_probes": []
            }

        with open(variant_file, "w", encoding="utf-8") as f:
            json.dump(variant_data, f, indent=2)

        results_by_variant[variant] = variant_data

    # Generate comparison summary file
    comparison_file = out_dir / "comparison.json"
    comparison_data = {
        "timestamp": now_iso,
        "dataset": "SecureLoRA Controlled Canary Memorization Benchmark",
        "dataset_id": "canary_synthetic_benchmark",
        "base_model": base_model_name,
        "sample_count": len(canaries),
        "seed": seed,
        "threat_model": {
            "attack_vector": "Black-box prefix probing completion",
            "prompt_protocol": "Held-out context prompt (secret string strictly excluded from prompt)",
            "distinction": "PII detection precision/recall evaluates input redaction. Canary extraction evaluates whether trained model weights memorize sensitive secrets.",
            "dp_separation_note": "Differential privacy (ε, δ) bounds are mathematical guarantees from DP-SGD and are reported separately from empirical extraction rates."
        },
        "variants": results_by_variant
    }

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    logger.info("Saved privacy evaluation results to %s", out_dir)
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Privacy & Memorization Evaluation Pipeline")
    parser.add_argument("--dataset", type=str, default="canary_synthetic_benchmark", help="Dataset identifier")
    parser.add_argument("--split", type=str, default="test", help="Dataset split")
    parser.add_argument("--samples", type=int, default=30, help="Number of evaluation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--model", type=str, default="JackFram/llama-68m", help="Base model identifier")
    parser.add_argument("--adapter", type=str, default="secure_lora_adapter", help="Adapter identifier or path")
    parser.add_argument("--output-dir", type=str, default=str(PRIVACY_OUT_DIR), help="Output directory")
    parser.add_argument("--allow-direct-loading", action="store_true", help="Allow direct loading of HuggingFace models for evaluation")

    args = parser.parse_args()

    results = evaluate_privacy_pipeline(
        dataset_id=args.dataset,
        split=args.split,
        samples=args.samples,
        seed=args.seed,
        base_model_name=args.model,
        adapter_id=args.adapter,
        output_dir=Path(args.output_dir),
        allow_direct_loading=args.allow_direct_loading
    )
    print(f"\n Privacy evaluation completed. Reports generated at -> {args.output_dir}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
