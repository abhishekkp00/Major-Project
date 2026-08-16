"""
chat_engine.py
==============
Privacy-Preserving Chat & Inference Engine for SecureLoRA.

Uses the canonical ModelRegistry and inference_service to execute
Base Model vs PEFT Adapter Model generation.
No fake, deterministic, or analytics responses when model inference is requested.
If SecureLoRA model is unavailable, returns status="MODEL_UNAVAILABLE".
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from src.orchestrator.model_registry import model_registry
from src.orchestrator.inference_service import (
    generate_base,
    generate_securelora,
    compare_base_and_securelora,
)

logger = logging.getLogger("secure_lora.chat_engine")


def register_model(model, tokenizer, model_name: str, adapter_id: str = "secure_lora_adapter", deployment_id: str = "verified_deployment"):
    """Forwards model registration to global model_registry."""
    model_registry.register(
        base_model=getattr(model, "base_model", model),
        peft_model=model,
        tokenizer=tokenizer,
        base_model_name=model_name,
        adapter_id=adapter_id,
        deployment_id=deployment_id,
        deployment_status="VERIFIED"
    )


def get_registered_model():
    info = model_registry.get_info()
    return info["peft_model"], info["tokenizer"], info["base_model_name"]


def generate_with_securelora_model(
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Executes actual PyTorch model generation using the canonical inference_service.
    NO analytics fallback.
    Returns explicit MODEL_UNAVAILABLE status if no verified PEFT model is loaded.
    """
    config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
    }
    return compare_base_and_securelora(prompt, config)


def answer_question(
    question: str,
    records: Optional[List[Dict[str, Any]]] = None
) -> Tuple[str, str, bool]:
    """
    Backwards compatible interface for chat prompts.
    Delegates to compare_base_and_securelora. Returns MODEL_UNAVAILABLE if model is missing.
    """
    if not question.strip():
        return "Please ask a question about your dataset.", "SAFE", False

    if not model_registry.is_verified():
        return "MODEL_UNAVAILABLE: SecureLoRA deployment must be verified first.", "UNAVAILABLE", True

    res = compare_base_and_securelora(question)
    if res.get("status") == "SUCCESS":
        return res["securelora_output"], "SAFE", False

    return "MODEL_UNAVAILABLE: SecureLoRA deployment must be verified first.", "UNAVAILABLE", True
