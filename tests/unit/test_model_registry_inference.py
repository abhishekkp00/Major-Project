"""
test_model_registry_inference.py
=================================
Unit tests for SecureLoRA ModelRegistry, PEFT inference, and PII evaluation.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch

from src.orchestrator.model_registry import ModelRegistry, model_registry
from src.orchestrator.chat_engine import generate_with_securelora_model, answer_question


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, return_tensors="pt", **kwargs):
        # Return mock input ids tensor
        return {"input_ids": torch.tensor([[101, 102, 103]])}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids == [201, 202]:
            return "Base Model response text with john.doe@acme.corp."
        elif token_ids == [301, 302]:
            return "SecureLoRA response text with masked entities."
        return "Decoded text output."


class DummyBaseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def generate(self, input_ids=None, **kwargs):
        # Return base model tokens [input_ids (3), generated (2)]
        return torch.tensor([[101, 102, 103, 201, 202]])


class DummyPeftModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.param = torch.nn.Parameter(torch.ones(1))
        self.adapter_enabled = True

    def generate(self, input_ids=None, **kwargs):
        if not self.adapter_enabled:
            return torch.tensor([[101, 102, 103, 201, 202]])
        return torch.tensor([[101, 102, 103, 301, 302]])

    def disable_adapter(self):
        class DisableContext:
            def __init__(ctx, outer):
                ctx.outer = outer
            def __enter__(ctx):
                ctx.outer.adapter_enabled = False
            def __exit__(ctx, *args):
                ctx.outer.adapter_enabled = True
        return DisableContext(self)


class TestModelRegistryAndInference(unittest.TestCase):

    def setUp(self):
        model_registry.clear()

    def tearDown(self):
        model_registry.clear()

    def test_1_peft_model_registration(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="sshleifer/tiny-gpt2",
            adapter_name="test_job_123",
            deployment_id="dep_456"
        )
        self.assertTrue(model_registry.is_verified())

    def test_2_tokenizer_registration(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        info = model_registry.get_info()
        self.assertIs(info["tokenizer"], tokenizer)

    def test_3_model_retrieval(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
            adapter_name="adapter-xyz"
        )
        info = model_registry.get_info()
        self.assertIs(info["base_model"], base_model)
        self.assertIs(info["peft_model"], peft_model)
        self.assertEqual(info["adapter_name"], "adapter-xyz")

    def test_4_generation_uses_peft_model(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("SecureLoRA response text", res["securelora_output"])

    def test_5_base_model_generation_uses_base_model(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertIn("Base Model response text", res["base_output"])

    def test_6_both_outputs_are_independently_generated(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertNotEqual(res["base_output"], res["securelora_output"])

    def test_7_analytics_fallback_only_when_model_unavailable(self):
        # When model is not registered:
        res = generate_with_securelora_model("Test prompt")
        self.assertEqual(res["status"], "MODEL_UNAVAILABLE")
        self.assertFalse(res["adapter_active"])

    def test_8_pii_evaluation_happens_after_generation(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertIn("base_pii", res)
        self.assertIn("securelora_pii", res)
        self.assertIsInstance(res["base_pii"]["count"], int)
        self.assertIsInstance(res["securelora_pii"]["count"], int)

    def test_9_adapter_active_reflects_actual_peft_state(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertTrue(res["adapter_active"])

    def test_10_model_verified_reflects_deployment_verification(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        res = generate_with_securelora_model("Test prompt")
        self.assertTrue(res["model_verified"])
        self.assertEqual(res["model_info"]["status"], "VERIFIED")


if __name__ == "__main__":
    unittest.main()
