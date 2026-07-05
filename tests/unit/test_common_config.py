import os
import pytest
from src.common.config_loader import config


def test_config_loader_defaults():
    # Verify defaults are defined
    assert config.model_name == "JackFram/llama-68m"
    assert config.device_salt is not None


def test_config_loader_env_override():
    os.environ["SECURE_LORA_MODEL_NAME"] = "test-model-override"
    try:
        # Re-import or force reload is not strictly needed if we just check the property
        # depending on implementation. But we can check if it returns env override:
        assert config.model_name == "test-model-override"
    finally:
        del os.environ["SECURE_LORA_MODEL_NAME"]
