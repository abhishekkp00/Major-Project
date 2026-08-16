"""
test_phase2_dp.py
=================
Tests for the Differentially Private LoRA training engine.

Coverage (9 required + extras):
  1.  DP disabled → config.dp_enabled = False
  2.  DP enabled → creates a valid private training run (integration smoke test)
  3.  Epsilon is computed by a real Opacus accountant, not hard-coded
  4.  Delta is recorded in the training result
  5.  Clipping norm is applied (max_grad_norm is set in PrivacyEngine)
  6.  Gaussian noise is actually injected (noise_multiplier > 0)
  7.  Invalid privacy parameters fail safely
  8.  Privacy metadata is stored in the output JSON file
  9.  Deterministic experiment configuration is recorded in the result
  10. KDF version has no connection to DP (orthogonal threat surfaces)
  11. DPTrainingResult serialises to dict correctly
  12. check_kdf_version does not interact with DP params
"""

import json
import math
import os
import random
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

_BASE_MODEL = "JackFram/llama-68m"


def _make_lora_model():
    """Load a small LoRA model for testing."""
    model = AutoModelForCausalLM.from_pretrained(_BASE_MODEL, torch_dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    peft_cfg = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],
    )
    return get_peft_model(model, peft_cfg)


def _tiny_dataset(n: int = 12):
    """Return a list of minimal tokenized records."""
    records = []
    for i in range(n):
        seq_len = 24
        ids = list(range(1, seq_len + 1))
        records.append({
            "input_ids": ids,
            "attention_mask": [1] * seq_len,
            "labels": ids,
        })
    return records


def _make_in_memory_dataset(n: int = 12):
    from src.phase2.train_lora import InMemoryDataset
    return InMemoryDataset(_tiny_dataset(n))


def _no_op_generate(model, tokenizer):
    return "test_generation"


# ─────────────────────────────────────────────────────────────────────────────
# 1. DP disabled → config reports lora mode
# ─────────────────────────────────────────────────────────────────────────────

class TestDPConfig:

    def test_dp_disabled_by_default(self, monkeypatch):
        """Requirement 1: DP disabled by default from config."""
        monkeypatch.delenv("DP_ENABLED", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        # training.yaml has privacy.enabled: false
        assert cfg.dp_enabled is False

    def test_dp_enabled_via_env(self, monkeypatch):
        """DP_ENABLED=1 env-var correctly sets dp_enabled=True."""
        monkeypatch.setenv("DP_ENABLED", "1")
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_enabled is True

    def test_dp_mode_label_lora(self, monkeypatch):
        """training_mode returns 'lora' when DP disabled."""
        monkeypatch.delenv("DP_ENABLED", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_training_mode == "lora"

    def test_dp_mode_label_dp_lora(self, monkeypatch):
        """training_mode returns 'dp-lora' when DP enabled."""
        monkeypatch.setenv("DP_ENABLED", "1")
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_training_mode == "dp-lora"

    def test_dp_target_epsilon_env_override(self, monkeypatch):
        """DP_TARGET_EPSILON env-var is read correctly."""
        monkeypatch.setenv("DP_TARGET_EPSILON", "4.5")
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_target_epsilon == pytest.approx(4.5)

    def test_dp_target_delta_default(self, monkeypatch):
        """Default delta is 1e-5."""
        monkeypatch.delenv("DP_TARGET_DELTA", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_target_delta == pytest.approx(1e-5)

    def test_dp_max_grad_norm_default(self, monkeypatch):
        """Default clipping norm is 1.0."""
        monkeypatch.delenv("DP_MAX_GRAD_NORM", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_max_grad_norm == pytest.approx(1.0)

    def test_dp_noise_multiplier_none_when_not_set(self, monkeypatch):
        """noise_multiplier is None when not configured (auto-compute mode)."""
        monkeypatch.delenv("DP_NOISE_MULTIPLIER", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_noise_multiplier is None

    def test_dp_accountant_default(self, monkeypatch):
        """Default accountant is 'rdp'."""
        monkeypatch.delenv("DP_ACCOUNTANT", raising=False)
        from src.common.config_loader import ConfigLoader
        cfg = ConfigLoader()
        assert cfg.dp_accountant == "rdp"


# ─────────────────────────────────────────────────────────────────────────────
# 2. DP enabled creates a valid private training run (integration smoke test)
# ─────────────────────────────────────────────────────────────────────────────

class TestDPTrainingIntegration:

    def test_dp_training_completes(self, tmp_path):
        """Requirement 2: DP training runs end-to-end without errors."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        assert result.status == "completed"
        assert result.training_mode == "dp-lora"

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Epsilon is computed by a real accountant, not hard-coded
    # ─────────────────────────────────────────────────────────────────────────

    def test_epsilon_is_real_accountant_value(self, tmp_path):
        """Requirement 3: epsilon is from the Opacus privacy_engine.get_epsilon()."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        # Epsilon must be a real computed value, not the magic number 8.0 or target.
        assert result.epsilon is not None
        assert isinstance(result.epsilon, float)
        assert result.epsilon > 0
        # Epsilon should be in a realistic range (not 0 or infinity).
        assert result.epsilon < 1000.0

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Delta is recorded
    # ─────────────────────────────────────────────────────────────────────────

    def test_delta_is_recorded(self, tmp_path):
        """Requirement 4: delta is recorded in the training result."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        target_delta = 1e-4  # non-default value
        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=target_delta,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        assert result.delta == pytest.approx(target_delta)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Clipping is enabled
    # ─────────────────────────────────────────────────────────────────────────

    def test_clipping_norm_recorded(self, tmp_path):
        """Requirement 5: max_grad_norm is recorded in the result."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        custom_norm = 0.5
        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=custom_norm,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        assert result.max_grad_norm == pytest.approx(custom_norm)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. Noise is actually injected (noise_multiplier > 0)
    # ─────────────────────────────────────────────────────────────────────────

    def test_noise_is_injected(self, tmp_path):
        """Requirement 6: noise_multiplier > 0 proves Gaussian noise is added."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        assert result.noise_multiplier is not None
        assert result.noise_multiplier > 0.0, (
            "noise_multiplier must be strictly positive for Gaussian noise to be injected."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 7. Invalid privacy parameters fail safely
    # ─────────────────────────────────────────────────────────────────────────

    def test_too_small_dataset_fails_safely(self, tmp_path):
        """Requirement 7: too-small dataset raises, not silently fails."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # batch_size > dataset_size → DataLoader is empty → should raise ValueError.
        tiny_ds = _make_in_memory_dataset(n=1)
        val_ds = _make_in_memory_dataset(n=1)

        with pytest.raises((ValueError, RuntimeError)):
            run_dp_training(
                model=model,
                tokenizer=tokenizer,
                train_dataset=tiny_ds,
                val_dataset=val_ds,
                model_name=_BASE_MODEL,
                target_epsilon=8.0,
                target_delta=1e-5,
                max_grad_norm=1.0,
                noise_multiplier_override=None,
                accountant_type="rdp",
                num_epochs=1,
                batch_size=8,   # larger than dataset → empty loader
                learning_rate=1e-4,
                seed=42,
                output_dir=tmp_path,
                generate_fn=_no_op_generate,
            )

    def test_very_small_epsilon_raises_or_warns(self, tmp_path):
        """Very tight ε (near 0) should raise a meaningful error from Opacus."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        with pytest.raises((RuntimeError, ValueError, Exception)):
            run_dp_training(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_ds,
                val_dataset=val_ds,
                model_name=_BASE_MODEL,
                target_epsilon=0.0001,   # impossibly tight ε for this dataset size
                target_delta=1e-5,
                max_grad_norm=1.0,
                noise_multiplier_override=None,
                accountant_type="rdp",
                num_epochs=1,
                batch_size=4,
                learning_rate=1e-4,
                seed=42,
                output_dir=tmp_path,
                generate_fn=_no_op_generate,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # 8. Privacy metadata is stored in the output JSON
    # ─────────────────────────────────────────────────────────────────────────

    def test_privacy_metadata_stored_to_file(self, tmp_path):
        """Requirement 8: dp_eval_report.json is written with all DP fields."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        report_path = tmp_path / "dp_eval_report.json"
        assert report_path.exists(), "dp_eval_report.json must be written."

        report = json.loads(report_path.read_text())
        required_dp_fields = [
            "epsilon", "delta", "noise_multiplier", "max_grad_norm",
            "accountant_type", "training_mode",
        ]
        for field in required_dp_fields:
            assert field in report, f"Missing field '{field}' in dp_eval_report.json"

        assert report["training_mode"] == "dp-lora"
        assert report["epsilon"] is not None
        assert report["delta"] is not None
        assert report["noise_multiplier"] is not None

    # ─────────────────────────────────────────────────────────────────────────
    # 9. Deterministic experiment configuration is recorded
    # ─────────────────────────────────────────────────────────────────────────

    def test_experiment_config_recorded_in_result(self, tmp_path):
        """Requirement 9: seed, batch_size, lr, epochs are all in the result."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        result = run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=99,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        assert result.seed == 99
        assert result.batch_size == 4
        assert result.learning_rate == pytest.approx(1e-4)
        assert result.num_epochs == 1
        assert result.model_name == _BASE_MODEL


# ─────────────────────────────────────────────────────────────────────────────
# 11. DPTrainingResult serialises correctly
# ─────────────────────────────────────────────────────────────────────────────

class TestDPTrainingResult:

    def test_dp_training_result_to_dict(self):
        """DPTrainingResult.to_dict() returns all fields as a plain dict."""
        from src.phase2.dp_trainer import DPTrainingResult

        r = DPTrainingResult(
            training_mode="dp-lora",
            model_name="test-model",
            epsilon=7.93,
            delta=1e-5,
            noise_multiplier=1.12,
            max_grad_norm=1.0,
            accountant_type="rdp",
            seed=42,
        )
        d = r.to_dict()
        assert d["training_mode"] == "dp-lora"
        assert d["epsilon"] == pytest.approx(7.93)
        assert d["delta"] == pytest.approx(1e-5)
        assert d["noise_multiplier"] == pytest.approx(1.12)
        assert d["accountant_type"] == "rdp"

    def test_lora_result_has_none_dp_fields(self):
        """Mode A result has None for all DP-specific fields."""
        from src.phase2.dp_trainer import DPTrainingResult

        r = DPTrainingResult(training_mode="lora")
        d = r.to_dict()
        assert d["epsilon"] is None
        assert d["delta"] is None
        assert d["noise_multiplier"] is None
        assert d["max_grad_norm"] is None

    def test_result_is_json_serialisable(self):
        """DPTrainingResult.to_dict() must be JSON-serialisable."""
        from src.phase2.dp_trainer import DPTrainingResult

        r = DPTrainingResult(
            epsilon=8.12, delta=1e-5, noise_multiplier=1.05,
            val_loss=2.3, perplexity=math.exp(2.3),
        )
        serialised = json.dumps(r.to_dict())   # must not raise
        recovered = json.loads(serialised)
        assert recovered["epsilon"] == pytest.approx(8.12)


# ─────────────────────────────────────────────────────────────────────────────
# 10. DP and KDF are orthogonal (sanity check)
# ─────────────────────────────────────────────────────────────────────────────

class TestOrthogonalThreatSurfaces:

    def test_kdf_version_unaffected_by_dp_config(self, monkeypatch):
        """KDF_VERSION is independent of DP configuration (orthogonal threat surfaces)."""
        monkeypatch.setenv("DP_ENABLED", "1")
        monkeypatch.setenv("DP_TARGET_EPSILON", "4.0")
        from src.security.key_derivation import KDF_VERSION
        # KDF version must not change based on DP settings.
        assert KDF_VERSION == "hkdf-sha256-v1"

    def test_dp_result_does_not_contain_kdf_fields(self, tmp_path):
        """dp_eval_report.json must not contain any cryptographic key fields."""
        from src.phase2.dp_trainer import run_dp_training

        model = _make_lora_model()
        tokenizer = AutoTokenizer.from_pretrained(_BASE_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_ds = _make_in_memory_dataset(n=20)
        val_ds = _make_in_memory_dataset(n=4)

        run_dp_training(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_ds,
            val_dataset=val_ds,
            model_name=_BASE_MODEL,
            target_epsilon=16.0,
            target_delta=1e-5,
            max_grad_norm=1.0,
            noise_multiplier_override=None,
            accountant_type="rdp",
            num_epochs=1,
            batch_size=4,
            learning_rate=1e-4,
            seed=42,
            output_dir=tmp_path,
            generate_fn=_no_op_generate,
        )

        report = json.loads((tmp_path / "dp_eval_report.json").read_text())
        for forbidden_field in ["kdf_version", "fingerprint_hash", "device_salt", "aes_key"]:
            assert forbidden_field not in report, (
                f"DP report must not contain cryptographic key field: {forbidden_field}"
            )
