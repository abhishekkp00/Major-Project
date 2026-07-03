import pytest
from pathlib import Path

from phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter
from peft import PeftModel

@pytest.fixture
def base_model_name():
    return "JackFram/llama-68m"

@pytest.fixture
def local_adapter_dir() -> Path:
    # Use the existing Phase 2 outputs/final_adapter if it exists,
    # or return Path to it for real testing.
    return Path("outputs/final_adapter")

def test_load_base_model_success(base_model_name):
    model, tokenizer = load_base_model_and_tokenizer(base_model_name)
    assert model is not None
    assert tokenizer is not None

def test_load_peft_adapter_success(base_model_name, local_adapter_dir):
    if not local_adapter_dir.exists() or not (local_adapter_dir / "adapter_config.json").exists():
        pytest.skip("Phase 2 final_adapter files not present, skipping real PEFT load test.")
    
    base_model, tokenizer = load_base_model_and_tokenizer(base_model_name)
    peft_model = load_peft_adapter(base_model, local_adapter_dir)
    assert isinstance(peft_model, PeftModel)
