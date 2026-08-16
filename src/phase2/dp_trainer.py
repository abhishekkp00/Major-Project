"""
dp_trainer.py
=============
Differentially Private LoRA training engine for SecureLoRA (Phase 2, Mode B).

Privacy Mechanism
-----------------
This module wraps the standard LoRA fine-tuning loop with Opacus
PrivacyEngine to provide (ε, δ)-differential privacy via:

  1. Per-example gradient computation (GradSampleModule).
  2. Per-example gradient clipping at norm C (= max_grad_norm):
         g̃_i = g_i · min(1, C / ‖g_i‖₂)

  3. Gaussian noise injection scaled by σ (= noise_multiplier):
         g̃ = (1/|B|) · (Σᵢ g̃_i + N(0, σ²C²I))

  4. Privacy accounting via Rényi DP composition (or PRV/f-DP):
         ε = accountant.get_epsilon(delta)

Privacy accountant
------------------
  - "rdp" (default): Moments accountant / Rényi DP (Abadi et al. 2016,
    arXiv:1607.00133). Tight for Gaussian mechanism.
  - "prv": PRV / f-DP accountant (Gopi et al. 2021, arXiv:2106.02848).
    Tighter for large numbers of steps.

IMPORTANT
---------
DP-LoRA addresses *training-data privacy* (membership inference resistance).
It does NOT protect the adapter weights from theft — that is the purpose of
device-bound encryption in Phase 3/4. These are orthogonal threat surfaces.

DP-LoRA is an established research direction (Li et al. 2021, Yu et al. 2021).
Our contribution is the measured interaction of training-data privacy,
device-bound adapter protection, and deployment security.

Opacus compatibility notes
--------------------------
Opacus 1.x requires:
  - Batch size ≥ 2 (for gradient sampling).
  - No gradient_accumulation_steps > 1 in DP mode
    (accumulation breaks per-example gradient semantics).
  - DataLoader with batch_sampler (not sampler + drop_last tricks).
  - The model must pass opacus.validators.ModuleValidator.validate().

LoRA with Opacus
----------------
Only the trainable LoRA parameters (lora_A, lora_B matrices) receive
per-example gradients and noise.  The frozen base-model parameters are
excluded from the DP mechanism.  Opacus's GradSampleModule wraps only
parameters with requires_grad=True, so frozen params are never touched.
"""

from __future__ import annotations

import json
import logging
import math
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger("secure_lora.phase2.dp_trainer")


# ─────────────────────────────────────────────────────────────────────────────
# Result data class
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DPTrainingResult:
    """Structured result from a training run (Mode A or B)."""

    # Training mode
    training_mode: str = "lora"                # "lora" or "dp-lora"

    # Model info
    model_name: str = ""
    trainable_parameters: int = 0
    total_parameters: int = 0
    trainable_percent: float = 0.0

    # Dataset
    train_samples: int = 0
    val_samples: int = 0
    num_epochs: int = 0
    total_steps: int = 0

    # Losses / metrics
    train_loss: float = float("nan")
    val_loss: float = float("nan")
    perplexity: float = float("nan")

    # Throughput / resources
    training_duration_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    throughput_samples_per_sec: float = 0.0

    # DP-specific (None when training_mode == "lora")
    epsilon: Optional[float] = None
    delta: Optional[float] = None
    noise_multiplier: Optional[float] = None
    max_grad_norm: Optional[float] = None
    accountant_type: Optional[str] = None

    # Experiment reproducibility
    seed: int = 42
    batch_size: int = 2
    learning_rate: float = 2e-4

    # Generation samples
    pre_training_generation: str = ""
    post_training_generation: str = ""

    status: str = "completed"
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# DP-SGD training loop
# ─────────────────────────────────────────────────────────────────────────────

def _make_dp_dataloader(dataset, batch_size: int, seed: int) -> DataLoader:
    """
    Creates a DataLoader suitable for Opacus.

    Opacus requires a standard DataLoader with batch_size (not a custom
    batch_sampler).  We use drop_last=True to guarantee uniform batch sizes,
    which is required for the Poisson sampling approximation used by the
    privacy accountant.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,        # required: Opacus sampling theory assumes uniform batches
        generator=generator,
        num_workers=0,         # keep in-process for compatibility
        pin_memory=torch.cuda.is_available(),
    )


def _compute_loss(model, batch, device: str) -> torch.Tensor:
    """Runs a forward pass and returns the mean cross-entropy loss."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["labels"].to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    return outputs.loss


def run_dp_training(
    model,
    tokenizer,
    train_dataset,
    val_dataset,
    *,
    model_name: str,
    target_epsilon: float,
    target_delta: float,
    max_grad_norm: float,
    noise_multiplier_override: Optional[float],
    accountant_type: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output_dir: Path,
    generate_fn,
) -> DPTrainingResult:
    """
    DP-LoRA training loop (Mode B).

    Parameters
    ----------
    model:
        PEFT model with frozen base and trainable LoRA parameters.
    tokenizer:
        Corresponding tokenizer.
    train_dataset, val_dataset:
        InMemoryDataset instances (shared with Mode A for valid comparison).
    target_epsilon:
        Desired privacy budget ε.  The noise_multiplier is computed via
        binary search by Opacus to achieve this ε at the given δ.
    target_delta:
        Failure probability δ.  Recommended: 1/N where N = dataset size.
    max_grad_norm:
        Per-example gradient clipping norm C.
    noise_multiplier_override:
        If not None, bypasses target_epsilon and uses this σ directly.
        Post-hoc ε is still computed by the accountant.
    accountant_type:
        "rdp" or "prv".
    num_epochs, batch_size, learning_rate, seed:
        Identical to Mode A for controlled comparison.
    output_dir:
        Where to write the dp_eval_report.json.
    generate_fn:
        Callable(model, tokenizer) -> str for sample generation.
    """
    try:
        from opacus import PrivacyEngine
        from opacus.validators import ModuleValidator
    except ImportError as exc:
        raise ImportError(
            "Opacus is required for DP-LoRA. Install it with: pip install opacus>=1.4"
        ) from exc

    result = DPTrainingResult(
        training_mode="dp-lora",
        model_name=model_name,
        num_epochs=num_epochs,
        seed=seed,
        batch_size=batch_size,
        learning_rate=learning_rate,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        delta=target_delta,
        max_grad_norm=max_grad_norm,
        accountant_type=accountant_type,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # ── Count parameters ──────────────────────────────────────────────────────
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    result.trainable_parameters = trainable_params
    result.total_parameters = total_params
    result.trainable_percent = 100.0 * trainable_params / total_params if total_params else 0.0
    logger.info(
        "DP-LoRA: %d trainable / %d total params (%.4f%%).",
        trainable_params, total_params, result.trainable_percent
    )

    # ── Pre-training sample ───────────────────────────────────────────────────
    result.pre_training_generation = generate_fn(model, tokenizer)

    # ── Validate model for Opacus ─────────────────────────────────────────────
    # Opacus cannot wrap models with certain layer types (e.g. BatchNorm).
    # ModuleValidator.fix() replaces incompatible layers automatically.
    errors = ModuleValidator.validate(model, strict=False)
    if errors:
        logger.info("Opacus ModuleValidator found %d issues — attempting auto-fix.", len(errors))
        model = ModuleValidator.fix(model)
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            raise RuntimeError(
                f"Model cannot be made Opacus-compatible after auto-fix. "
                f"Remaining issues: {errors_after}"
            )

    # ── DataLoader ────────────────────────────────────────────────────────────
    train_loader = _make_dp_dataloader(train_dataset, batch_size, seed)

    if len(train_loader) == 0:
        raise ValueError(
            f"Training DataLoader is empty (dataset size {len(train_dataset)} < batch_size {batch_size}). "
            "Increase dataset size or decrease batch_size."
        )

    # ── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
    )

    # ── Privacy Engine ────────────────────────────────────────────────────────
    privacy_engine = PrivacyEngine(accountant=accountant_type)

    sample_rate = batch_size / len(train_dataset)
    num_steps = num_epochs * len(train_loader)
    result.total_steps = num_steps

    if noise_multiplier_override is not None:
        # User explicitly set noise_multiplier: wrap without epsilon targeting.
        logger.info(
            "DP-LoRA: using explicit noise_multiplier=%.4f (σ). "
            "Post-hoc ε will be computed by accountant.",
            noise_multiplier_override,
        )
        model, optimizer, train_loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=train_loader,
            noise_multiplier=noise_multiplier_override,
            max_grad_norm=max_grad_norm,
        )
        result.noise_multiplier = noise_multiplier_override

    else:
        # Auto-compute noise_multiplier from target_epsilon via binary search.
        logger.info(
            "DP-LoRA: computing noise_multiplier for target ε=%.2f, δ=%.2e "
            "(accountant=%s, steps=%d, sample_rate=%.6f).",
            target_epsilon, target_delta, accountant_type, num_steps, sample_rate,
        )
        try:
            model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                epochs=num_epochs,
                max_grad_norm=max_grad_norm,
            )
        except Exception as e:
            raise RuntimeError(
                f"Opacus could not compute noise_multiplier for ε={target_epsilon}, "
                f"δ={target_delta}. Dataset may be too small. Error: {e}"
            ) from e

        result.noise_multiplier = optimizer.noise_multiplier
        logger.info(
            "DP-LoRA: auto-selected noise_multiplier=%.4f.", result.noise_multiplier
        )

    logger.info(
        "DP-LoRA privacy engine attached. max_grad_norm=%.2f, noise_multiplier=%.4f.",
        max_grad_norm, result.noise_multiplier,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    model.train()
    peak_memory_bytes = 0
    start_time = time.perf_counter()
    step_losses = []

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            optimizer.zero_grad()
            loss = _compute_loss(model, batch, device)

            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("NaN/Inf loss at epoch %d — skipping batch.", epoch)
                continue

            loss.backward()   # Opacus computes per-example gradients here.
            optimizer.step()  # Clips, injects noise, and updates parameters.

            epoch_loss += loss.item()
            num_batches += 1
            step_losses.append(loss.item())

            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated()
                peak_memory_bytes = max(peak_memory_bytes, mem)

        avg_loss = epoch_loss / max(num_batches, 1)
        # Query privacy accountant for current ε after this epoch.
        eps_now = privacy_engine.get_epsilon(target_delta)
        logger.info(
            "Epoch %d/%d — loss=%.4f, ε=%.4f (δ=%.2e).",
            epoch, num_epochs, avg_loss, eps_now, target_delta,
        )

    duration = time.perf_counter() - start_time
    result.training_duration_seconds = round(duration, 2)
    result.train_loss = float(torch.tensor(step_losses).mean()) if step_losses else float("nan")
    result.peak_memory_mb = round(peak_memory_bytes / 1e6, 2)
    total_samples_processed = len(step_losses) * batch_size
    result.throughput_samples_per_sec = round(total_samples_processed / max(duration, 1e-9), 2)

    # ── Final privacy accountant query ────────────────────────────────────────
    epsilon_final = privacy_engine.get_epsilon(target_delta)
    result.epsilon = round(float(epsilon_final), 6)
    result.delta = target_delta
    logger.info(
        "DP-LoRA training complete. Final ε=%.4f (δ=%.2e), noise_multiplier=%.4f.",
        result.epsilon, result.delta, result.noise_multiplier,
    )

    # ── Validation loss ───────────────────────────────────────────────────────
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            try:
                loss = _compute_loss(model, batch, device)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    val_losses.append(loss.item())
            except Exception:
                pass

    if val_losses:
        result.val_loss = round(float(torch.tensor(val_losses).mean()), 6)
        result.perplexity = round(math.exp(min(result.val_loss, 20.0)), 4)
    else:
        result.val_loss = float("nan")
        result.perplexity = float("nan")

    result.post_training_generation = generate_fn(model, tokenizer)

    # ── Save DP report ────────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "dp_eval_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=4)
    logger.info("DP training report saved → %s", report_path)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Standard LoRA validation helper (shared evaluation path)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_validation_loss(model, val_dataset, batch_size: int, device: str) -> float:
    """Compute mean validation loss over val_dataset. Shared by Mode A and B."""
    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    losses = []
    with torch.no_grad():
        for batch in val_loader:
            try:
                loss = _compute_loss(model, batch, device)
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    losses.append(loss.item())
            except Exception:
                pass
    return float(torch.tensor(losses).mean()) if losses else float("nan")
