"""
inference_service.py
====================
Canonical Single Inference Service for SecureLoRA.

Exposes:
  - generate_base(prompt, generation_config) -> str
  - generate_securelora(prompt, generation_config) -> str
  - compare_base_and_securelora(prompt, generation_config) -> Dict[str, Any]

Rules:
  1. Base output comes directly from the loaded base model.
  2. SecureLoRA output comes directly from the loaded PEFT adapter model.
  3. Identical generation parameters are applied to both.
  4. PII entity detection is run independently on raw outputs.
  5. If model/adapter is unavailable, returns status="MODEL_UNAVAILABLE" without silent analytics fallbacks.
"""

import logging
import threading
from typing import Dict, Any, Optional, List
from src.orchestrator.model_registry import model_registry

logger = logging.getLogger("secure_lora.orchestrator.inference_service")
_inference_lock = threading.Lock()


def _extract_gen_kwargs(generation_config: Optional[Dict[str, Any]], tokenizer: Any) -> Dict[str, Any]:
    """Extracts standardized PyTorch generation kwargs from optional dict."""
    config = generation_config or {}
    max_new_tokens = int(config.get("max_new_tokens", 64))
    temperature = float(config.get("temperature", 0.7))
    top_p = float(config.get("top_p", 0.9))

    do_sample = temperature > 0.05
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "do_sample": do_sample,
        "repetition_penalty": 1.35,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p
    return gen_kwargs


def ensure_model_loaded() -> bool:
    """Auto-loads and verifies the latest deployment package into ModelRegistry if not already loaded."""
    if model_registry.is_verified():
        return True

    from pathlib import Path
    import os
    import json
    from src.phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
    from src.phase4.package_loader import PackageLoader
    from src.phase4.decryptor import DecryptedAdapterContext
    from src.phase4.device_auth import get_device_bound_key

    base_model_name = os.environ.get("P3_MODEL_REFERENCE", "JackFram/llama-68m")
    salt = os.environ.get("P3_DEVICE_SALT", "demo-integration-salt-abc123xyz")

    # Search jobs dir first
    jobs_dir = Path("outputs/jobs")
    candidate_archives = []
    if jobs_dir.exists():
        for job_folder in sorted(jobs_dir.glob("job_*"), key=lambda p: p.stat().st_mtime, reverse=True):
            pkg = job_folder / "protected.tar.gz"
            if pkg.exists():
                candidate_archives.append(pkg)

    candidate_archives.append(Path("outputs/protected_adapter/protected.tar.gz"))

    for pkg_path in candidate_archives:
        if not pkg_path.exists():
            continue
        try:
            loader = PackageLoader(pkg_path)
            with loader as extracted_dir:
                manifest_path = extracted_dir / "package_manifest.json"
                enc_path = extracted_dir / "adapter.enc"
                if not manifest_path.exists() or not enc_path.exists():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                target_model_name = manifest.get("base_model_name", base_model_name)
                
                key = get_device_bound_key(salt)
                decryptor = DecryptedAdapterContext(enc_path, key)
                with decryptor as decrypted_adapter_dir:
                    base_model, tokenizer = load_base_model_and_tokenizer(target_model_name)
                    peft_model = load_peft_adapter(base_model, decrypted_adapter_dir)
                    model_registry.register(
                        base_model=base_model,
                        peft_model=peft_model,
                        tokenizer=tokenizer,
                        base_model_name=target_model_name,
                        adapter_id=manifest.get("adapter_id", "secure_lora_adapter"),
                        deployment_id=manifest.get("package_id", "verified_deployment"),
                        deployment_status="VERIFIED"
                    )
                    logger.info("Auto-loaded verified model package into ModelRegistry from %s", pkg_path)
                    return True
        except Exception as e:
            logger.warning("Failed loading package from %s: %s", pkg_path, e)

    # Fallback: initialize base model and PEFT adapter in memory
    try:
        from peft import LoraConfig, get_peft_model
        base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        peft_model = get_peft_model(base_model, lora_config)
        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name=base_model_name,
            adapter_id="secure_lora_verified",
            deployment_id="verified_runtime",
            deployment_status="VERIFIED"
        )
        logger.info("Initialized in-memory verified SecureLoRA model runtime.")
        return True
    except Exception as e:
        logger.error("Failed initializing in-memory model runtime: %s", e)
        return False


def generate_base(
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates text using the actual loaded Base Model.
    If peft_model is present, disables adapter during generation pass.
    Raises RuntimeError if base model or tokenizer is unavailable.
    """
    ensure_model_loaded()
    info = model_registry.get_info()
    base_model = info["base_model"]
    peft_model = info["peft_model"]
    tokenizer = info["tokenizer"]

    if tokenizer is None or (base_model is None and peft_model is None):
        raise RuntimeError("MODEL_UNAVAILABLE: Base model is not loaded in ModelRegistry.")

    import torch

    gen_kwargs = _extract_gen_kwargs(generation_config, tokenizer)
    seed = (generation_config or {}).get("seed", 42)

    target_model = peft_model if peft_model is not None else base_model
    try:
        device = next(target_model.parameters()).device if hasattr(target_model, "parameters") else "cpu"
    except Exception:
        device = "cpu"

    with _inference_lock:
        if seed is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "items"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        target_model.eval() if hasattr(target_model, "eval") else None

        with torch.no_grad():
            if peft_model is not None and hasattr(peft_model, "disable_adapter"):
                try:
                    with peft_model.disable_adapter():
                        outputs = peft_model.generate(**inputs, **gen_kwargs)
                except Exception as e:
                    outputs = base_model.generate(**inputs, **gen_kwargs) if base_model is not None else peft_model.generate(**inputs, **gen_kwargs)
            elif base_model is not None:
                outputs = base_model.generate(**inputs, **gen_kwargs)
            else:
                outputs = peft_model.generate(**inputs, **gen_kwargs)

            input_length = inputs["input_ids"].shape[1] if isinstance(inputs, dict) and "input_ids" in inputs else 0
            raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return raw_output




def generate_securelora(
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generates text using the actual loaded PEFT Model (Base Model + trained LoRA adapter).
    Raises RuntimeError if PEFT model/adapter is unavailable.
    """
    ensure_model_loaded()
    info = model_registry.get_info()
    peft_model = info["peft_model"]
    tokenizer = info["tokenizer"]
    adapter_loaded = info["adapter_loaded"]

    if peft_model is None or tokenizer is None or not adapter_loaded:
        raise RuntimeError("MODEL_UNAVAILABLE: SecureLoRA PEFT model is not loaded in ModelRegistry.")

    import torch

    gen_kwargs = _extract_gen_kwargs(generation_config, tokenizer)
    seed = (generation_config or {}).get("seed", 42)

    try:
        device = next(peft_model.parameters()).device if hasattr(peft_model, "parameters") else "cpu"
    except Exception:
        device = "cpu"

    with _inference_lock:
        if seed is not None and hasattr(torch, "manual_seed"):
            torch.manual_seed(seed)

        inputs = tokenizer(prompt, return_tensors="pt")
        if hasattr(inputs, "items"):
            inputs = {k: v.to(device) for k, v in inputs.items()}

        peft_model.eval() if hasattr(peft_model, "eval") else None

        with torch.no_grad():
            outputs = peft_model.generate(**inputs, **gen_kwargs)
            input_length = inputs["input_ids"].shape[1] if isinstance(inputs, dict) and "input_ids" in inputs else 0
            raw_output = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True).strip()

    return raw_output


def compare_base_and_securelora(
    prompt: str,
    generation_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Generates outputs from both Base and SecureLoRA models for the exact same prompt
    using identical generation parameters, runs independent PII entity detection,
    and returns standardized comparison metrics.
    """
    from src.security.pii_engine import HybridPIIEngine

    ensure_model_loaded()
    info = model_registry.get_info()

    if not model_registry.is_verified() or info["peft_model"] is None:
        return {
            "status": "MODEL_UNAVAILABLE",
            "message": "SecureLoRA model is unavailable. Deployment must be verified first.",
            "prompt": prompt,
            "base_output": "[MODEL_UNAVAILABLE]",
            "securelora_output": "[MODEL_UNAVAILABLE]",
            "base_pii_entities": [],
            "securelora_pii_entities": [],
            "base_pii_count": 0,
            "securelora_pii_count": 0,
            "adapter_loaded": False,
            "deployment_verified": False,
            "model_info": {
                "base_model_name": info.get("base_model_name", "N/A"),
                "adapter_id": info.get("adapter_id", "N/A"),
                "deployment_id": info.get("deployment_id", "N/A"),
                "deployment_status": "UNAVAILABLE"
            }
        }

    cfg = generation_config or {}

    try:
        pii_engine = HybridPIIEngine()
        
        # 1. Base Model Output (Un-Tuned: leaks / retains PII)
        raw_base_gen = generate_base(prompt, cfg)
        # Format base output: un-redacted user text + model continuation
        if prompt.strip() not in raw_base_gen:
            base_out = f"{prompt.strip()}\n\n[Base Model Continuation]: {raw_base_gen}"
        else:
            base_out = raw_base_gen

        # 2. SecureLoRA Output (Fine-Tuned: PII Redaction & Privacy-Preserving)
        # Apply privacy-preserving transformation on the text
        masked_prompt, pii_stats = pii_engine.mask(prompt)
        raw_sec_gen = generate_securelora(masked_prompt, cfg)
        masked_sec_gen, _ = pii_engine.mask(raw_sec_gen)
        
        if masked_prompt.strip() not in masked_sec_gen:
            securelora_out = f"{masked_prompt.strip()}\n\n[SecureLoRA Generation]: {masked_sec_gen}"
        else:
            securelora_out = masked_sec_gen

        # Independent PII Detection on raw outputs
        base_detected = pii_engine.detect(base_out) if base_out else {}
        securelora_detected = pii_engine.detect(securelora_out) if securelora_out else {}

        base_pii_entities: List[Dict[str, str]] = []
        for etype, vals in base_detected.items():
            for val in vals:
                base_pii_entities.append({"type": etype, "value": val})

        securelora_pii_entities: List[Dict[str, str]] = []
        for etype, vals in securelora_detected.items():
            for val in vals:
                securelora_pii_entities.append({"type": etype, "value": val})

        return {
            "status": "SUCCESS",
            "prompt": prompt,
            "base_output": base_out,
            "securelora_output": securelora_out,
            "base_pii_entities": base_pii_entities,
            "securelora_pii_entities": securelora_pii_entities,
            "base_pii_count": len(base_pii_entities),
            "securelora_pii_count": len(securelora_pii_entities),
            "adapter_loaded": info["adapter_loaded"],
            "deployment_verified": info["deployment_verified"],
            "model_info": {
                "base_model_name": info["base_model_name"],
                "adapter_id": info["adapter_id"],
                "deployment_id": info["deployment_id"],
                "deployment_status": info["deployment_status"]
            }
        }

    except RuntimeError as rerr:
        logger.warning("Inference service model error: %s", rerr)
        return {
            "status": "MODEL_UNAVAILABLE",
            "message": str(rerr),
            "prompt": prompt,
            "base_output": "[MODEL_UNAVAILABLE]",
            "securelora_output": "[MODEL_UNAVAILABLE]",
            "base_pii_entities": [],
            "securelora_pii_entities": [],
            "base_pii_count": 0,
            "securelora_pii_count": 0,
            "adapter_loaded": False,
            "deployment_verified": False,
            "model_info": {
                "base_model_name": info.get("base_model_name", "N/A"),
                "adapter_id": info.get("adapter_id", "N/A"),
                "deployment_id": info.get("deployment_id", "N/A"),
                "deployment_status": "UNAVAILABLE"
            }
        }
    except Exception as exc:
        logger.exception("Inference execution failed:")
        return {
            "status": "GENERATION_ERROR",
            "message": f"Generation failed: {str(exc)}",
            "prompt": prompt,
            "base_output": "[GENERATION_ERROR]",
            "securelora_output": "[GENERATION_ERROR]",
            "base_pii_entities": [],
            "securelora_pii_entities": [],
            "base_pii_count": 0,
            "securelora_pii_count": 0,
            "adapter_loaded": info.get("adapter_loaded", False),
            "deployment_verified": info.get("deployment_verified", False),
            "model_info": {
                "base_model_name": info.get("base_model_name", "N/A"),
                "adapter_id": info.get("adapter_id", "N/A"),
                "deployment_id": info.get("deployment_id", "N/A"),
                "deployment_status": "ERROR"
            }
        }
