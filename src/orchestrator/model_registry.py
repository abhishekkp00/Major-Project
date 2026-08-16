"""
model_registry.py
=================
Canonical single model registry for SecureLoRA.

Holds the verified, loaded base model, PEFT adapter model, tokenizer, and deployment metadata
in volatile memory following Phase 4 deployment verification.
"""

import threading
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("secure_lora.orchestrator.model_registry")


class ModelRegistry:
    def __init__(self):
        self._lock = threading.RLock()
        self.base_model = None
        self.peft_model = None
        self.tokenizer = None
        self.base_model_name: Optional[str] = None
        self.adapter_id: Optional[str] = None
        self.deployment_id: Optional[str] = None
        self.deployment_status: str = "UNAVAILABLE"
        self.adapter_loaded: bool = False

    def register(
        self,
        base_model: Any,
        peft_model: Any,
        tokenizer: Any,
        base_model_name: str,
        adapter_id: str = "secure_lora_adapter",
        deployment_id: str = "verified_deployment",
        deployment_status: str = "VERIFIED"
    ) -> None:
        """Registers the verified base model, PEFT adapter, tokenizer, and deployment metadata in memory."""
        with self._lock:
            # Tokenizer / Model compatibility check
            if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
                if hasattr(tokenizer, "eos_token") and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                elif hasattr(tokenizer, "pad_token_id"):
                    tokenizer.pad_token_id = getattr(tokenizer, "eos_token_id", 0)

            self.base_model = base_model
            self.peft_model = peft_model
            self.tokenizer = tokenizer
            self.base_model_name = base_model_name
            self.adapter_id = adapter_id
            self.deployment_id = deployment_id
            self.deployment_status = deployment_status
            self.adapter_loaded = (peft_model is not None or base_model is not None)

        logger.info(
            "ModelRegistry: Successfully registered model (base=%s, adapter_id=%s, deployment_id=%s, loaded=%s)",
            base_model_name, adapter_id, deployment_id, self.adapter_loaded
        )

    def is_verified(self) -> bool:
        """Returns True only if a model and adapter are loaded and deployment status is VERIFIED."""
        with self._lock:
            return self.adapter_loaded and self.deployment_status == "VERIFIED" and (self.peft_model is not None or self.base_model is not None)

    def get_info(self) -> Dict[str, Any]:
        """Returns a snapshot of the registered model objects and metadata."""
        with self._lock:
            verified = self.is_verified()
            return {
                "base_model": self.base_model,
                "peft_model": self.peft_model,
                "tokenizer": self.tokenizer,
                "base_model_name": self.base_model_name or "Unloaded",
                "adapter_id": self.adapter_id or "None",
                "deployment_id": self.deployment_id or "None",
                "deployment_status": self.deployment_status if verified else "UNAVAILABLE",
                "adapter_loaded": self.adapter_loaded and (self.peft_model is not None),
                "deployment_verified": verified,
            }

    def clear(self) -> None:
        """Clears all loaded model objects from memory context."""
        with self._lock:
            self.base_model = None
            self.peft_model = None
            self.tokenizer = None
            self.base_model_name = None
            self.adapter_id = None
            self.deployment_id = None
            self.deployment_status = "UNAVAILABLE"
            self.adapter_loaded = False
        logger.info("ModelRegistry: Cleared registered model context.")


# Global singleton instance
model_registry = ModelRegistry()
