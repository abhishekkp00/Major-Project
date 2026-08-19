"""
privacy_evaluator.py
====================
Reproducible Privacy Evaluation Pipeline for SecureLoRA (STEP 3).

Measures whether fine-tuned models (Base Model, Standard LoRA, DP-LoRA, SecureLoRA)
reduce sensitive PII/PHI information leakage in generated raw outputs.

Pipeline Sequence:
  MODEL GENERATION → RAW OUTPUT → PII DETECTION → METRICS → DISPLAY MASKING (OPTIONAL)

Results Output Directory:
  outputs/evaluation/privacy/
    ├── base_model.json
    ├── lora.json
    ├── dp_lora.json
    ├── securelora.json
    └── comparison.json

Command-line Reproducibility Arguments:
  --dataset      dataset ID (ai4privacy, synthea, synthetic)
  --split        dataset split (train, test)
  --samples      sample count
  --seed         random seed
  --model        base model identifier
  --adapter      adapter identifier
"""

import os
import sys
import json
import argparse
import logging
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

logger = logging.getLogger("secure_lora.evaluation.privacy_evaluator")

PRIVACY_OUT_DIR = _PROJECT_ROOT / "outputs" / "evaluation" / "privacy"


def _generate_model_output(
    variant: str,
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> Tuple[str, bool]:
    """
    Generates text from raw model instance for variant:
      - 'base_model'
      - 'lora'
      - 'dp_lora'
      - 'securelora'
    Returns tuple (raw_output_text, success_boolean).
    Does NOT sanitize output prior to leakage measurement.
    """
    info = model_registry.get_info()
    try:
        if variant == "base_model":
            if info.get("base_model") is None and info.get("peft_model") is None:
                return "", False
            out = generate_base(prompt, generation_config)
            return out, True
        elif variant == "securelora":
            if not info.get("adapter_loaded") or info.get("peft_model") is None:
                return "", False
            out = generate_securelora(prompt, generation_config)
            return out, True
        elif variant == "lora":
            adapter_type = str(info.get("adapter_type") or info.get("adapter_id") or "")
            if "dp" in adapter_type.lower():
                return "", False
            if not info.get("adapter_loaded") or info.get("peft_model") is None:
                return "", False
            out = generate_securelora(prompt, generation_config)
            return out, True
        elif variant == "dp_lora":
            adapter_type = str(info.get("adapter_type") or info.get("adapter_id") or "")
            dp_enabled = info.get("dp_enabled", False)
            if not dp_enabled and "dp" not in adapter_type.lower():
                return "", False
            if not info.get("adapter_loaded") or info.get("peft_model") is None:
                return "", False
            out = generate_securelora(prompt, generation_config)
            return out, True
        else:
            return "", False
    except Exception as exc:
        logger.warning("Model generation for variant '%s' failed: %s", variant, exc, exc_info=True)
        return "", False


def calculate_privacy_metrics(
    raw_outputs: List[str],
    ground_truth_records: Optional[List[Dict[str, Any]]] = None,
    ground_truth_available: bool = False
) -> Dict[str, Any]:
    """
    Calculates PII leakage rate, entity counts, types, PII-free response rate,
    and when ground_truth_available is True: precision, recall, F1, FPR, FNR.
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
        # MANDATORY: Run PII detection directly on raw_text (NO SANITIZATION BEFORE METRICS!)
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
    dataset_id: str = "ai4privacy",
    split: str = "test",
    samples: int = 50,
    seed: int = 42,
    base_model_name: str = "google/gemma-2b",
    adapter_id: str = "secure_lora_adapter",
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """Runs Privacy Evaluation Pipeline for all 4 variants and saves output files."""
    out_dir = Path(output_dir) if output_dir else PRIVACY_OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter = dataset_registry.get_dataset_adapter(dataset_id)
    records = adapter.load(subset_size=samples, split=split, seed=seed)
    meta = adapter.metadata()

    now_iso = datetime.now(timezone.utc).isoformat()
    variants = ["base_model", "lora", "dp_lora", "securelora"]
    results_by_variant = {}

    gen_config = {"temperature": 0.7, "max_new_tokens": 128, "seed": seed}

    for variant in variants:
        raw_outputs = []
        executed = True

        for rec in records:
            prompt = rec.get("input") or rec.get("instruction") or ""
            text_out, ok = _generate_model_output(variant, prompt, gen_config)
            if not ok:
                executed = False
                break
            raw_outputs.append(text_out)

        variant_file = out_dir / f"{variant}.json"

        if executed and raw_outputs:
            metrics = calculate_privacy_metrics(
                raw_outputs,
                ground_truth_records=records,
                ground_truth_available=adapter.ground_truth_available
            )
            variant_data = {
                "dataset": adapter.dataset_name,
                "dataset_id": adapter.dataset_id,
                "model": base_model_name,
                "adapter": adapter_id if variant != "base_model" else "none",
                "seed": seed,
                "sample_count": len(records),
                "timestamp": now_iso,
                "status": "EXECUTED",
                "ground_truth_available": adapter.ground_truth_available,
                "configuration": {
                    "split": split,
                    "subset_size": len(records),
                    "seed": seed,
                    "temperature": gen_config["temperature"],
                    "max_new_tokens": gen_config["max_new_tokens"]
                },
                "metrics": metrics
            }
        else:
            # STRICT NO FABRICATION: Set status = NOT_EXECUTED when experiment was not executed
            variant_data = {
                "dataset": adapter.dataset_name,
                "dataset_id": adapter.dataset_id,
                "model": base_model_name,
                "adapter": adapter_id if variant != "base_model" else "none",
                "seed": seed,
                "sample_count": len(records),
                "timestamp": now_iso,
                "status": "NOT_EXECUTED",
                "reason": f"Model or adapter instance for variant '{variant}' was not loaded or executed.",
                "ground_truth_available": adapter.ground_truth_available,
                "configuration": {
                    "split": split,
                    "subset_size": len(records),
                    "seed": seed
                },
                "metrics": None
            }

        with open(variant_file, "w", encoding="utf-8") as f:
            json.dump(variant_data, f, indent=2)

        results_by_variant[variant] = variant_data

    # Generate comparison summary file
    comparison_file = out_dir / "comparison.json"
    comparison_data = {
        "timestamp": now_iso,
        "dataset": adapter.dataset_name,
        "dataset_id": adapter.dataset_id,
        "base_model": base_model_name,
        "sample_count": len(records),
        "seed": seed,
        "variants": results_by_variant
    }

    with open(comparison_file, "w", encoding="utf-8") as f:
        json.dump(comparison_data, f, indent=2)

    logger.info("Saved privacy evaluation results to %s", out_dir)
    return comparison_data


def main():
    parser = argparse.ArgumentParser(description="SecureLoRA Privacy Evaluation Pipeline (STEP 3)")
    parser.add_argument("--dataset", type=str, default="ai4privacy", help="Dataset ID (ai4privacy, synthea, synthetic)")
    parser.add_argument("--split", type=str, default="test", help="Dataset split (train, test)")
    parser.add_argument("--samples", type=int, default=50, help="Number of evaluation samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--model", type=str, default="google/gemma-2b", help="Base model identifier")
    parser.add_argument("--adapter", type=str, default="secure_lora_adapter", help="Adapter identifier")
    parser.add_argument("--output-dir", type=str, default=str(PRIVACY_OUT_DIR), help="Output directory")

    args = parser.parse_args()

    results = evaluate_privacy_pipeline(
        dataset_id=args.dataset,
        split=args.split,
        samples=args.samples,
        seed=args.seed,
        base_model_name=args.model,
        adapter_id=args.adapter,
        output_dir=Path(args.output_dir)
    )
    print(f"\n Privacy evaluation completed. Reports generated at -> {args.output_dir}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
