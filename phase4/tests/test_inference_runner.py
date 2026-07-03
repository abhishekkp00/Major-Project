import pytest
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from phase4.inference_runner import mask_sensitive_output, generate_text, run_side_by_side_inference
from phase4.adapter_loader import load_base_model_and_tokenizer, load_peft_adapter

@pytest.fixture(scope="module")
def model_and_tokenizer():
    model_name = "JackFram/llama-68m"
    base_model, tokenizer = load_base_model_and_tokenizer(model_name)
    return base_model, tokenizer

def test_mask_sensitive_output():
    # Test email masking
    assert mask_sensitive_output("My email is admin@corporate.com") == "My email is [MASKED_EMAIL]"
    
    # Test SSN masking
    assert mask_sensitive_output("My SSN is 123-45-6789.") == "My SSN is [MASKED_SSN]."
    
    # Test API key/secret masking
    assert mask_sensitive_output("Here is the api-key: 'xyz123secret'") == "Here is the api-key: [MASKED_SECRET]"
    assert mask_sensitive_output("Here is the secret = \"somepasswd12\"") == "Here is the secret: [MASKED_SECRET]"

def test_generate_text_success(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    prompt = "Secure computing requires"
    output = generate_text(model, tokenizer, prompt, max_new_tokens=10, do_sample=False)
    assert isinstance(output, str)
    assert len(output) > 0

def test_side_by_side_inference(model_and_tokenizer):
    base_model, tokenizer = model_and_tokenizer
    local_adapter_dir = Path("outputs/final_adapter")
    
    if not local_adapter_dir.exists() or not (local_adapter_dir / "adapter_config.json").exists():
        pytest.skip("Phase 2 final_adapter files not present, skipping PEFT load/inference test.")
        
    peft_model = load_peft_adapter(base_model, local_adapter_dir)
    prompt = "What is security?"
    
    result = run_side_by_side_inference(base_model, peft_model, tokenizer, prompt, max_new_tokens=10)
    assert "prompt" in result
    assert "base_output" in result
    assert "peft_output" in result
    assert "adapter_active" in result
