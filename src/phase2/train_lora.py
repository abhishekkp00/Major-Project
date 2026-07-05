import os
import sys
import json
import logging
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
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

from src.common.config_loader import config
from src.utils.logging_utils import setup_logging
from src.utils.checkpoint_utils import find_latest_checkpoint

logger = setup_logging()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class InMemoryDataset(PyTorchDataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.data[idx]["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(self.data[idx]["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(self.data[idx]["labels"], dtype=torch.long)
        }


class SecureCheckpointCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        from src.utils.checkpoint_utils import rotate_checkpoints
        rotate_checkpoints(Path(args.output_dir), max_to_keep=2)


def generate_sample(model, tokenizer, prompt_text=None) -> str:
    if not prompt_text:
        prompt_text = os.getenv("SECURE_LORA_SAMPLE_PROMPT", "Mask all Personally Identifiable Information (PII) in the text.\nInput: Nombre: Blaise. Edad: 25.")

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
            top_k=50
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Response: " in full_output:
        return full_output.split("Response: ")[1].strip()
    return full_output.strip()


def run_training():
    logger.info("Initializing secure fine-tuning pipeline...")

    try:
        config.validate_phase2()
    except Exception as e:
        logger.error("Workspace validation failed: %s", e)
        sys.exit(1)

    set_seed(config.seed)

    # Resolve AES decryption key from runtime environment
    from src.phase1.cli import resolve_key
    try:
        key = resolve_key()
    except SystemExit:
        logger.error("Failed to load decryption key.")
        sys.exit(1)

    from src.security import decrypted_temporary_file
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

    logger.info("Loading base model %s...", config.model_name)
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.float32
        )
    except Exception as e:
        logger.error("Model load error: %s", e)
        sys.exit(1)

    logger.info("Tokenizing dataset in-memory...")
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

            # Label mask: set instruction tokens to -100 so they don't contribute to loss calculation
            labels = [-100] * prompt_len + tokenized_full["input_ids"][prompt_len:]
            labels = labels[:len(tokenized_full["input_ids"])]

            tokenized_data.append({
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": labels
            })
        elif "text" in record:
            full_text = record["text"] + tokenizer.eos_token
            tokenized = tokenizer(full_text, truncation=True, max_length=config.max_seq_length)
            tokenized_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy()
            })

    if not tokenized_data:
        logger.error("No valid dataset records found.")
        sys.exit(1)

    random.shuffle(tokenized_data)
    split_idx = max(1, int(len(tokenized_data) * 0.9))
    train_dataset = InMemoryDataset(tokenized_data[:split_idx])
    val_dataset = InMemoryDataset(tokenized_data[split_idx:])
    logger.info("Dataset split: %d train, %d validation samples.", len(train_dataset), len(val_dataset))

    logger.info("Injecting LoRA adapters...")
    for param in model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias=config.lora_bias,
        task_type=TaskType.CAUSAL_LM,
        target_modules=config.target_modules
    )

    model = get_peft_model(model, peft_config)
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info("Trainable parameters: %s / %s (%s%%)", f"{trainable_params:,}", f"{all_param:,}", f"{100 * trainable_params / all_param:.4f}")

    logger.info("Generating pre-training baseline text output...")
    pre_gen = generate_sample(model, tokenizer)
    logger.info("Baseline Output: %s", pre_gen)

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
        report_to="none"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        pad_to_multiple_of=8,
        return_tensors="pt",
        padding=True
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        callbacks=[SecureCheckpointCallback()]
    )

    latest_checkpoint = find_latest_checkpoint(config.checkpoint_dir)

    logger.info("Starting training loop...")
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
        logger.info("Fine-tuning completed successfully.")
    except Exception as e:
        logger.error("Training loop failed: %s", e)
        sys.exit(1)

    logger.info("Saving PEFT adapter to %s...", config.lora_output_dir)
    model.save_pretrained(config.lora_output_dir)
    tokenizer.save_pretrained(config.lora_output_dir)

    logger.info("Running validation evaluations...")
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", float("inf"))
    perplexity = np.exp(val_loss) if val_loss < 20 else float("inf")

    post_gen = generate_sample(model, tokenizer)
    logger.info("Post-Training Sanity Output: %s", post_gen)

    report = {
        "model_name": config.model_name,
        "trainable_parameters": trainable_params,
        "all_parameters": all_param,
        "trainable_percent": 100 * trainable_params / all_param,
        "validation_loss": val_loss,
        "perplexity": perplexity,
        "pre_training_generation": pre_gen,
        "post_training_generation": post_gen,
        "status": "completed"
    }

    # Save evaluation report to standardized output directory
    eval_report_dir = Path("outputs/evaluation")
    eval_report_dir.mkdir(parents=True, exist_ok=True)
    report_file = eval_report_dir / "eval_report.json"

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)
    # Also save to legacy path for backward compatibility
    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    logger.info("Evaluation report saved to %s", report_file)


if __name__ == "__main__":
    run_training()
