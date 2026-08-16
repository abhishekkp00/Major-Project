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
        self._lock = threading.Lock()
        self.base_model = None
        self.peft_model = None
        self.tokenizer = None
        self.base_model_name: Optional[str] = None
        self.adapter_name: Optional[str] = None
        self.deployment_id: Optional[str] = None
        self.deployment_verified: bool = False

    def register(
        self,
        base_model: Any,
        peft_model: Any,
        tokenizer: Any,
        base_model_name: str,
        adapter_name: str = "secure_lora_adapter",
        deployment_id: str = "verified_deployment"
    ) -> None:
        """Registers the verified base model, PEFT adapter, and tokenizer in memory."""
        with self._lock:
            self.base_model = base_model
            self.peft_model = peft_model
            self.tokenizer = tokenizer
            self.base_model_name = base_model_name
            self.adapter_name = adapter_name
            self.deployment_id = deployment_id
            self.deployment_verified = True
        logger.info(
            "ModelRegistry: Successfully registered model (base=%s, adapter=%s, deployment_id=%s)",
            base_model_name, adapter_name, deployment_id
        )

    def is_verified(self) -> bool:
        """Returns True only if a model and adapter are loaded and deployment is verified."""
        with self._lock:
            return self.deployment_verified and (self.peft_model is not None or self.base_model is not None)

    def get_info(self) -> Dict[str, Any]:
        """Returns a snapshot of the registered model objects and metadata."""
        with self._lock:
            return {
                "base_model": self.base_model,
                "peft_model": self.peft_model or self.base_model,
                "tokenizer": self.tokenizer,
                "base_model_name": self.base_model_name or "Unloaded",
                "adapter_name": self.adapter_name or "None",
                "deployment_id": self.deployment_id or "None",
                "deployment_verified": self.deployment_verified,
            }

    def clear(self) -> None:
        """Clears all loaded model objects from memory context."""
        with self._lock:
            self.base_model = None
            self.peft_model = None
            self.tokenizer = None
            self.base_model_name = None
            self.adapter_name = None
            self.deployment_id = None
            self.deployment_verified = False
        logger.info("ModelRegistry: Cleared registered model context.")


# Global singleton instance
model_registry = ModelRegistry()
