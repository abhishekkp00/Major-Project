"""
train_lora.py
=============
Phase 2 — Secure LoRA Fine-Tuning Pipeline.

Supports two training modes (controlled by config.dp_enabled or DP_ENABLED env-var):

  Mode A — Standard LoRA
    Dataset → LoRA → HuggingFace Trainer → Adapter
    No privacy mechanism.  Baseline for utility measurement.

  Mode B — DP-LoRA
    Dataset → PII sanitization (Phase 1) → LoRA → DP-SGD (Opacus) → Adapter
    Per-example gradient clipping + Gaussian noise injection.
    Privacy budget tracked by Rényi DP or PRV accountant.

Both modes use the same:
  - dataset, seed, model, train/val split, epochs, learning rate,
    evaluation pipeline

so results are scientifically comparable.

IMPORTANT
---------
DP-LoRA addresses training-data privacy (membership inference resistance).
It does NOT protect the adapter from theft — that is the role of device-bound
encryption in Phase 3/4.  These are orthogonal threat surfaces.

DP-LoRA is an established research direction (Li et al. 2021, Yu et al. 2021).
Our research contribution is the measured interaction of training-data privacy,
device-bound adapter protection, and deployment security.
"""

import os
import sys
import json
import logging
import math
import time
import random
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset as PyTorchDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

from src.common.config_loader import config
from src.utils.logging_utils import setup_logging
from src.utils.checkpoint_utils import find_latest_checkpoint

logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class InMemoryDataset(PyTorchDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Trainer callbacks (Mode A only — HuggingFace Trainer)
# ─────────────────────────────────────────────────────────────────────────────

class SecureCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        from src.utils.checkpoint_utils import rotate_checkpoints
        rotate_checkpoints(Path(args.output_dir), max_to_keep=2)


class SecureProgressCallback(TrainerCallback):
    def __init__(self, progress_file_path: Path):
        self.progress_file_path = progress_file_path
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            from datetime import datetime, timezone
            progress_data = {
                "current_step": state.global_step,
                "total_steps": state.max_steps,
                "epoch": state.epoch,
                "learning_rate": logs.get("learning_rate"),
                "loss": logs.get("loss"),
                "eval_loss": logs.get("eval_loss"),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if "loss" in logs or "eval_loss" in logs:
                self.history.append(progress_data)

            try:
                temp_path = self.progress_file_path.with_suffix(".tmp")
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "current_step": state.global_step,
                            "total_steps": state.max_steps,
                            "epoch": state.epoch,
                            "history": self.history,
                        },
                        f, indent=4,
                    )
                temp_path.replace(self.progress_file_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared sample generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_sample(model, tokenizer, prompt_text: str = None) -> str:
    if not prompt_text:
        prompt_text = os.getenv(
            "SECURE_LORA_SAMPLE_PROMPT",
            "Mask all Personally Identifiable Information (PII) in the text.\n"
            "Input: Nombre: Blaise. Edad: 25.",
        )

    model.eval()
    device = next(model.parameters()).device
    formatted_prompt = f"Instruction: {prompt_text}\nResponse: "
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Response: " in full_output:
        return full_output.split("Response: ")[1].strip()
    return full_output.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading helper (shared)
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_tokenize(tokenizer) -> list:
    """Decrypts the training dataset and returns tokenized records."""
    from src.phase1.cli import resolve_key
    from src.security import decrypted_temporary_file

    try:
        key = resolve_key()
    except SystemExit:
        logger.error("Failed to load decryption key.")
        sys.exit(1)

    raw_records = []
    logger.info("Decrypting training dataset in-memory...")
    try:
        with decrypted_temporary_file(config.encrypted_dataset_path, key) as temp_path:
            if not temp_path.exists():
                raise FileNotFoundError("Temporary decrypted file creation failed.")
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped:
                        raw_records.append(json.loads(stripped))
        logger.info("Decrypted temporary files cleared successfully.")
    except Exception as e:
        logger.error("Failed to load dataset: %s", e)
        sys.exit(1)

    tokenized_data = []
    for record in raw_records:
        if "instruction" in record and "output" in record:
            prompt = f"Instruction: {record['instruction']}\n"
            if record.get("input"):
                prompt += f"Input: {record['input']}\n"
            prompt += "Response: "
            response = record["output"]

            full_text = prompt + response + tokenizer.eos_token
            tokenized_full = tokenizer(full_text, truncation=True, max_length=config.max_seq_length)
            tokenized_prompt = tokenizer(prompt, truncation=True, max_length=config.max_seq_length)
            prompt_len = len(tokenized_prompt["input_ids"])

            labels = [-100] * prompt_len + tokenized_full["input_ids"][prompt_len:]
            labels = labels[: len(tokenized_full["input_ids"])]

            tokenized_data.append({
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": labels,
            })
        elif "text" in record:
            full_text = record["text"] + tokenizer.eos_token
            tokenized = tokenizer(full_text, truncation=True, max_length=config.max_seq_length)
            tokenized_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy(),
            })

    return tokenized_data


# ─────────────────────────────────────────────────────────────────────────────
# Mode A — Standard LoRA with HuggingFace Trainer
# ─────────────────────────────────────────────────────────────────────────────

def _run_standard_lora(model, tokenizer, train_dataset, val_dataset):
    """Mode A: standard HuggingFace Trainer-based LoRA fine-tuning."""
    import time
    start = time.perf_counter()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    training_args = TrainingArguments(
        output_dir=str(config.checkpoint_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        seed=config.seed,
        remove_unused_columns=False,
        report_to="none",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer, model=model, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )

    callbacks = [SecureCheckpointCallback()]
    progress_file = os.getenv("SECURE_LORA_PROGRESS_FILE")
    if progress_file:
        callbacks.append(SecureProgressCallback(Path(progress_file)))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)
    trainer.train(resume_from_checkpoint=latest_checkpoint)

    duration = time.perf_counter() - start
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", float("nan"))
    perplexity = math.exp(val_loss) if val_loss < 20 else float("inf")

    peak_mem_mb = 0.0
    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated() / 1e6

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    total_steps = trainer.state.global_step

    return {
        "training_mode": "lora",
        "model_name": config.model_name,
        "trainable_parameters": trainable_params,
        "total_parameters": all_params,
        "trainable_percent": 100.0 * trainable_params / all_params if all_params else 0.0,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_epochs": config.num_epochs,
        "total_steps": total_steps,
        "train_loss": float(trainer.state.log_history[-1].get("train_loss", float("nan"))),
        "val_loss": round(float(val_loss), 6),
        "perplexity": round(float(perplexity), 4),
        "training_duration_seconds": round(duration, 2),
        "peak_memory_mb": round(peak_mem_mb, 2),
        "throughput_samples_per_sec": round(len(train_dataset) * config.num_epochs / max(duration, 1e-9), 2),
        "epsilon": None,
        "delta": None,
        "noise_multiplier": None,
        "max_grad_norm": None,
        "accountant_type": None,
        "seed": config.seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "status": "completed",
    }, model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point (called by orchestrator and CLI)
# ─────────────────────────────────────────────────────────────────────────────

def run_training(dp_enabled: bool = None) -> dict:
    """
    Main training entry point.

    Parameters
    ----------
    dp_enabled:
        Override whether DP is active.  If None, reads from config.dp_enabled.

    Returns
    -------
    dict: structured training result (also written to outputs/evaluation/).
    """
    logger.info("Initializing secure fine-tuning pipeline...")

    try:
        config.validate_phase2()
    except Exception as e:
        logger.error("Workspace validation failed: %s", e)
        sys.exit(1)

    if dp_enabled is None:
        dp_enabled = config.dp_enabled

    mode_label = "DP-LoRA (Mode B)" if dp_enabled else "Standard LoRA (Mode A)"
    logger.info("Training mode: %s", mode_label)

    set_seed(config.seed)

    # ── Load model ────────────────────────────────────────────────────────────
    logger.info("Loading base model %s...", config.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float32
        )
    except Exception as e:
        logger.error("Model load error: %s", e)
        sys.exit(1)

    # ── Load dataset ──────────────────────────────────────────────────────────
    logger.info("Tokenizing dataset in-memory...")
    tokenized_data = _load_and_tokenize(tokenizer)

    if not tokenized_data:
        logger.error("No valid dataset records found.")
        sys.exit(1)

    random.shuffle(tokenized_data)
    split_idx = max(1, int(len(tokenized_data) * 0.9))
    train_dataset = InMemoryDataset(tokenized_data[:split_idx])
    val_dataset = InMemoryDataset(tokenized_data[split_idx:])
    logger.info(
        "Dataset split: %d train, %d validation samples.",
        len(train_dataset), len(val_dataset),
    )

    # ── Inject LoRA adapters (shared by both modes) ───────────────────────────
    logger.info("Injecting LoRA adapters...")
    for param in model.parameters():
        param.requires_grad = False  # freeze base model

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules,
    )
    model = get_peft_model(model, peft_config)

    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(
        "Trainable parameters: %s / %s (%.4f%%)",
        f"{trainable_params:,}", f"{all_param:,}",
        100 * trainable_params / all_param,
    )

    logger.info("Generating pre-training baseline sample...")
    pre_gen = generate_sample(model, tokenizer)
    logger.info("Baseline output: %s", pre_gen)

    # ── Branch: Mode A or Mode B ──────────────────────────────────────────────
    eval_report_dir = Path("outputs/evaluation")
    eval_report_dir.mkdir(parents=True, exist_ok=True)

    if dp_enabled:
        # ── Mode B: DP-LoRA via Opacus ────────────────────────────────────────
        from src.phase2.dp_trainer import run_dp_training

        try:
            result = run_dp_training(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                model_name=config.model_name,
                target_epsilon=config.dp_target_epsilon,
                target_delta=config.dp_target_delta,
                max_grad_norm=config.dp_max_grad_norm,
                noise_multiplier_override=config.dp_noise_multiplier,
                accountant_type=config.dp_accountant,
                num_epochs=config.num_epochs,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                seed=config.seed,
                output_dir=eval_report_dir,
                generate_fn=generate_sample,
            )
            report = result.to_dict()
            report["pre_training_generation"] = pre_gen
            logger.info(
                "DP-LoRA complete. Final ε=%.4f, δ=%.2e, noise_multiplier=%.4f.",
                result.epsilon, result.delta, result.noise_multiplier,
            )
        except Exception as e:
            logger.error("DP-LoRA training failed: %s", e)
            raise

        # Save the PEFT adapter (Opacus does not call model.save_pretrained automatically)
        # Unwrap the PrivacyEngine wrapper before saving.
        try:
            from opacus import GradSampleModule
            unwrapped = model._module if isinstance(model, GradSampleModule) else model
            unwrapped.save_pretrained(config.lora_output_dir)
            tokenizer.save_pretrained(config.lora_output_dir)
        except Exception as e:
            logger.warning("Could not save adapter in standard format: %s. Saving state dict.", e)
            import pickle
            lora_state_dict = {
                k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k
            }
            pkl_path = Path(config.lora_output_dir) / "adapter_model.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(lora_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        # ── Mode A: Standard LoRA ─────────────────────────────────────────────
        report, model = _run_standard_lora(model, tokenizer, train_dataset, val_dataset)
        report["pre_training_generation"] = pre_gen

        post_gen = generate_sample(model, tokenizer)
        report["post_training_generation"] = post_gen

        logger.info("Saving PEFT adapter to %s...", config.lora_output_dir)
        model.save_pretrained(config.lora_output_dir)
        tokenizer.save_pretrained(config.lora_output_dir)

        import pickle
        lora_state_dict = {k: v.cpu() for k, v in model.state_dict().items() if "lora_" in k}
        pkl_path = Path(config.lora_output_dir) / "adapter_model.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(lora_state_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── Write evaluation report ───────────────────────────────────────────────
    report_file = eval_report_dir / "eval_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation report saved → %s", report_file)
    logger.info("Fine-tuning completed. Mode: %s", mode_label)
    return report


if __name__ == "__main__":
    run_training()
