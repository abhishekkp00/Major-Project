"""
phase4/inference_runner.py
---------------------------
Manages secure inference execution on the base model and PEFT model.
Includes comparative (side-by-side) inference runners and output-masking
utilities for diagnostics to prevent leaking sensitive information.
"""

import re
import logging
from typing import Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

# Basic patterns for masking sensitive information in diagnostics (PII, credentials, keys)
SENSITIVE_PATTERNS = [
    (re.compile(r"(api[-_]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"]+['\"]", re.IGNORECASE), r"\1: [MASKED_SECRET]"),
    (re.compile(r"[\w\.-]+@[\w\.-]+\.\w+"), "[MASKED_EMAIL]"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[MASKED_SSN]"),
]

def mask_sensitive_output(text: str) -> str:
    """Redacts typical sensitive patterns (email, SSN, API secrets) from the text."""
    masked = text
    for pattern, replacement in SENSITIVE_PATTERNS:
        masked = pattern.sub(replacement, masked)
    return masked

def generate_text(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True
) -> str:
    """Generates text from the model given a prompt."""
    logger.info("Executing model generation. Input prompt len: %d characters", len(prompt))
    
    # Enforce evaluation mode and no-gradient context
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt")
        # Ensure model handles inputs on the correct device (CPU/GPU)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Decode only the generated tokens (exclude prompt if desired, but standard is decode whole)
        generated_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
    masked_response = mask_sensitive_output(response)
    logger.info("Generation complete. Output response len: %d characters", len(masked_response))
    return masked_response

def run_side_by_side_inference(
    base_model: AutoModelForCausalLM,
    peft_model: PeftModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 64
) -> Dict[str, Any]:
    """
    Runs inference on the base model (without adapter) and peft model (with adapter)
    and returns a comparison dictionary.
    """
    logger.info("Running side-by-side comparative inference.")
    
    # 1. Base Model Inference (deactivating adapter temporarily if loading PEFT directly,
    # or running the raw base model instance).
    # Since we have the separate base_model instance, we can generate from it directly.
    base_output = generate_text(base_model, tokenizer, prompt, max_new_tokens, do_sample=False)
    
    # 2. PEFT Model Inference
    peft_output = generate_text(peft_model, tokenizer, prompt, max_new_tokens, do_sample=False)
    
    # Compare outputs to verify modification
    outputs_differ = base_output != peft_output
    
    logger.info("Side-by-side inference completed. Outputs differ: %s", outputs_differ)
    
    return {
        "prompt": prompt,
        "base_output": base_output,
        "peft_output": peft_output,
        "adapter_active": outputs_differ
    }
