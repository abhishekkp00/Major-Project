"""
test_model_registry_inference.py
=================================
Unit tests for SecureLoRA ModelRegistry, PEFT inference service, and PII evaluation.

Proves:
  1. Base model is actually used for BASE generation.
  2. PEFT model is actually used for SECURELORA generation.
  3. adapter_loaded is True ONLY when adapter exists in registry.
  4. Analytics fallback cannot replace valid model inference (returns MODEL_UNAVAILABLE).
  5. Tokenizer/model compatibility is checked (pad token fallback).
  6. Failed model loading produces MODEL_UNAVAILABLE.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch

from src.orchestrator.model_registry import ModelRegistry, model_registry
from src.orchestrator.inference_service import (
    generate_base,
    generate_securelora,
    compare_base_and_securelora,
)
from src.orchestrator.chat_engine import answer_question, generate_with_securelora_model


class DummyTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.pad_token_id = None
        self.eos_token_id = 2

    def __call__(self, text, return_tensors="pt", **kwargs):
        return {"input_ids": torch.tensor([[101, 102, 103]])}

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if token_ids == [201, 202]:
            return "Base Model response containing john.doe@acme.corp."
        elif token_ids == [301, 302]:
            return "SecureLoRA response with redacted medical notes."
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
        self.ml_patcher = patch("src.security.pii_engine.get_ml_engine")
        self.mock_ml = self.ml_patcher.start()
        mock_instance = MagicMock()
        mock_instance.extract_entities.return_value = []
        self.mock_ml.return_value = mock_instance

    def tearDown(self):
        self.ml_patcher.stop()
        model_registry.clear()

    def test_1_base_model_is_actually_used_for_base(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base-68m",
            adapter_id="adapter-v1",
            deployment_id="dep-101"
        )
        base_out = generate_base("What is the patient condition?")
        self.assertIn("Base Model response", base_out)

    def test_2_peft_model_is_actually_used_for_securelora(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base-68m",
            adapter_id="adapter-v1",
            deployment_id="dep-101"
        )
        sec_out = generate_securelora("What is the patient condition?")
        self.assertIn("SecureLoRA response", sec_out)
        self.assertNotEqual(generate_base("Test"), sec_out)

    def test_3_adapter_loaded_is_true_only_when_adapter_exists(self):
        info_unloaded = model_registry.get_info()
        self.assertFalse(info_unloaded["adapter_loaded"])

        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
            adapter_id="adapter-xyz",
            deployment_id="dep-123"
        )
        info_loaded = model_registry.get_info()
        self.assertTrue(info_loaded["adapter_loaded"])
        self.assertTrue(info_loaded["deployment_verified"])

    def test_4_analytics_fallback_cannot_replace_valid_model_inference(self):
        # Unloaded model must return MODEL_UNAVAILABLE status without synthetic analytics
        res = compare_base_and_securelora("Summarize dataset records.")
        self.assertEqual(res["status"], "MODEL_UNAVAILABLE")
        self.assertEqual(res["base_output"], "[MODEL_UNAVAILABLE]")
        self.assertEqual(res["securelora_output"], "[MODEL_UNAVAILABLE]")

        ans_res, status, blocked = answer_question("Summarize dataset records.")
        self.assertIn("MODEL_UNAVAILABLE", ans_res)
        self.assertEqual(status, "UNAVAILABLE")

    def test_5_tokenizer_model_compatibility_is_checked(self):
        tokenizer = DummyTokenizer()
        self.assertIsNone(tokenizer.pad_token)

        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
        )
        # Verify pad token was set to eos_token for compatibility
        self.assertEqual(tokenizer.pad_token, "<eos>")

    def test_6_failed_model_loading_produces_model_unavailable(self):
        model_registry.clear()
        res = generate_with_securelora_model("Test prompt")
        self.assertEqual(res["status"], "MODEL_UNAVAILABLE")
        self.assertFalse(res["adapter_loaded"])
        self.assertFalse(res["deployment_verified"])

    def test_7_side_by_side_comparison_with_pii_evaluation(self):
        base_model = DummyBaseModel()
        peft_model = DummyPeftModel(base_model)
        tokenizer = DummyTokenizer()

        model_registry.register(
            base_model=base_model,
            peft_model=peft_model,
            tokenizer=tokenizer,
            base_model_name="test-base",
            adapter_id="test_adapter",
            deployment_id="dep_001"
        )
        res = compare_base_and_securelora("Check PII")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(res["adapter_loaded"])
        self.assertTrue(res["deployment_verified"])
        self.assertIn("john.doe@acme.corp", res["base_output"])
        self.assertGreater(res["base_pii_count"], 0)
        self.assertIsInstance(res["base_pii_entities"], list)
        self.assertIsInstance(res["securelora_pii_entities"], list)


if __name__ == "__main__":
    unittest.main()
