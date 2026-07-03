"""
phase4/adapter_loader.py
-------------------------
Loads the base LLM model and merges/wraps it with the decrypted LoRA adapter.
Ensures that the PEFT module is loaded correctly.
"""

import logging
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

def load_base_model_and_tokenizer(model_name_or_path: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the causal LM base model and tokenizer from Hugging Face or local path.
    """
    logger.info("Loading base model and tokenizer: %s", model_name_or_path)
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        # Load base model on CPU for verification/testing portability
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
        logger.info("Base model and tokenizer loaded successfully.")
        return base_model, tokenizer
    except Exception as e:
        logger.error("Failed to load base model or tokenizer: %s", e)
        raise RuntimeError(f"Base model load failure: {e}") from e

def load_peft_adapter(base_model: AutoModelForCausalLM, adapter_dir: Path) -> PeftModel:
    """
    Loads the decrypted PEFT adapter weights into the base model.
    """
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Decrypted adapter directory not found: {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json missing from: {adapter_dir}")

    logger.info("Loading PEFT adapter into base model from: %s", adapter_dir)
    try:
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            low_cpu_mem_usage=True
        )
        logger.info("PEFT adapter successfully loaded and applied.")
        return peft_model
    except Exception as e:
        logger.error("Failed to load PEFT adapter: %s", e)
        raise RuntimeError(f"PEFT adapter load failure: {e}") from e
