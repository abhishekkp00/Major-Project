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

from config import TrainingConfig
from utils.logging_utils import setup_logging
from utils.checkpoint_utils import find_latest_checkpoint

# Initialize Logging
logger = setup_logging()

def set_seed(seed: int):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class InMemoryDataset(PyTorchDataset):
    """Simple in-memory dataset wrapper for pre-tokenized features."""
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
    """Callback to run secure checkpoint rotation after each save event."""
    def on_save(self, args, state, control, **kwargs):
        from utils.checkpoint_utils import rotate_checkpoints
        rotate_checkpoints(Path(args.output_dir), max_to_keep=2)

def generate_sample(model, tokenizer, prompt_text="What is the model name?") -> str:
    """Utility to generate text sample from the model for visual verification."""
    model.eval()
    device = next(model.parameters()).device
    
    # Format instruction prompt
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
    # Extract only response if possible
    if "Response: " in full_output:
        return full_output.split("Response: ")[1].strip()
    return full_output.strip()

def run_training():
    logger.info("Starting Phase 2: Secure LoRA Fine-Tuning Pipeline...")
    
    # 1. Validate Config & Workspace Boundaries
    try:
        TrainingConfig.validate()
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return
        
    set_seed(TrainingConfig.SEED)
    
    # 2. Ingest and Tokenize Decrypted Data In-Memory
    # Load secret key
    from secure_lora.cli import resolve_key
    try:
        key = resolve_key()
    except SystemExit:
        logger.error("Could not resolve decryption key. Training aborted.")
        return
        
    from secure_lora.security import decrypted_temporary_file
    raw_records = []
    
    logger.info("Step 1: Ephemerally decrypting dataset for in-memory tokenization...")
    try:
        with decrypted_temporary_file(TrainingConfig.ENCRYPTED_DATASET_PATH, key) as temp_path:
            if not temp_path.exists():
                raise FileNotFoundError("Temp decrypted dataset file could not be generated.")
            with open(temp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        raw_records.append(json.loads(line_str))
        logger.info("Step 2: Plaintext temp dataset file successfully shredded and unlinked.")
    except Exception as e:
        logger.error(f"Failed to securely process dataset: {e}")
        return
        
    # 3. Load Tokenizer & Base LLM
    logger.info(f"Step 3: Loading base LLM: {TrainingConfig.MODEL_NAME}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TrainingConfig.MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForCausalLM.from_pretrained(
            TrainingConfig.MODEL_NAME,
            torch_dtype=torch.float32  # Standard float32 for CPU correctness
        )
    except Exception as e:
        logger.error(f"Failed to load base LLM or Tokenizer: {e}")
        return
        
    # 4. Tokenize In-Memory Records & Mask Prompts
    logger.info("Step 4: Tokenizing ingested dataset records in-memory...")
    tokenized_data = []
    for record in raw_records:
        if "instruction" in record and "output" in record:
            # Build prompt structure
            prompt = f"Instruction: {record['instruction']}\n"
            if record.get("input"):
                prompt += f"Input: {record['input']}\n"
            prompt += "Response: "
            response = record["output"]
            
            full_text = prompt + response + tokenizer.eos_token
            tokenized_full = tokenizer(full_text, truncation=True, max_length=TrainingConfig.MAX_SEQ_LENGTH)
            
            tokenized_prompt = tokenizer(prompt, truncation=True, max_length=TrainingConfig.MAX_SEQ_LENGTH)
            prompt_len = len(tokenized_prompt["input_ids"])
            
            # Mask instruction loss
            labels = [-100] * prompt_len + tokenized_full["input_ids"][prompt_len:]
            labels = labels[:len(tokenized_full["input_ids"])]
            
            tokenized_data.append({
                "input_ids": tokenized_full["input_ids"],
                "attention_mask": tokenized_full["attention_mask"],
                "labels": labels
            })
        elif "text" in record:
            full_text = record["text"] + tokenizer.eos_token
            tokenized = tokenizer(full_text, truncation=True, max_length=TrainingConfig.MAX_SEQ_LENGTH)
            tokenized_data.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy()
            })
            
    if not tokenized_data:
        logger.error("No valid training samples parsed. Aborting training.")
        return
        
    # Split train/validation sets (90/10 split)
    random.shuffle(tokenized_data)
    split_idx = max(1, int(len(tokenized_data) * 0.9))
    train_data = tokenized_data[:split_idx]
    val_data = tokenized_data[split_idx:]
    
    train_dataset = InMemoryDataset(train_data)
    val_dataset = InMemoryDataset(val_data)
    logger.info(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    
    # 5. Lock Base Model & Apply PEFT LoRA Configuration
    logger.info("Step 5: Injecting PEFT LoRA Adapters...")
    # Freeze all base model weights
    for param in model.parameters():
        param.requires_grad = False
        
    peft_config = LoraConfig(
        r=TrainingConfig.LORA_R,
        lora_alpha=TrainingConfig.LORA_ALPHA,
        lora_dropout=TrainingConfig.LORA_DROPOUT,
        bias=TrainingConfig.LORA_BIAS,
        task_type=TaskType.CAUSAL_LM,
        target_modules=TrainingConfig.TARGET_MODULES
    )
    
    model = get_peft_model(model, peft_config)
    
    trainable_params, all_param = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable_params:,} | All parameters: {all_param:,} | Trainable %: {100 * trainable_params / all_param:.4f}%")
    
    # 6. Capture Pre-Training Generation Quality
    logger.info("Capturing baseline model text generation...")
    pre_gen = generate_sample(model, tokenizer)
    logger.info(f"Baseline Output: {pre_gen}")
    
    # 7. Configure HF Trainer with Resumption Support
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    training_args = TrainingArguments(
        output_dir=str(TrainingConfig.CHECKPOINT_DIR),
        per_device_train_batch_size=TrainingConfig.BATCH_SIZE,
        per_device_eval_batch_size=TrainingConfig.BATCH_SIZE,
        gradient_accumulation_steps=TrainingConfig.GRADIENT_ACCUMULATION_STEPS,
        learning_rate=TrainingConfig.LEARNING_RATE,
        num_train_epochs=TrainingConfig.NUM_EPOCHS,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        seed=TrainingConfig.SEED,
        remove_unused_columns=False,
        report_to="none"  # Keep metrics local, no external telemetry leaks
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
    
    # Check for latest checkpoints to resume from crashes
    latest_checkpoint = find_latest_checkpoint(TrainingConfig.CHECKPOINT_DIR)
    
    # 8. Run Training Loop
    logger.info("Step 6: Executing model fine-tuning...")
    try:
        trainer.train(resume_from_checkpoint=latest_checkpoint)
        logger.info("Fine-tuning completed successfully.")
    except Exception as e:
        logger.error(f"Training loop failed or interrupted: {e}")
        sys.exit(1)
        
    # 9. Save Final Adapter Weights Only
    logger.info(f"Step 7: Saving final PEFT adapter to {TrainingConfig.OUTPUT_DIR}...")
    model.save_pretrained(TrainingConfig.OUTPUT_DIR)
    tokenizer.save_pretrained(TrainingConfig.OUTPUT_DIR)
    
    # 10. Run Post-Training Evaluation and Generate Report
    logger.info("Step 8: Executing validation and sanity check generation...")
    eval_results = trainer.evaluate()
    val_loss = eval_results.get("eval_loss", float("inf"))
    perplexity = np.exp(val_loss) if val_loss < 20 else float("inf")
    
    post_gen = generate_sample(model, tokenizer)
    logger.info(f"Post-Training Sanity Output: {post_gen}")
    
    report = {
        "model_name": TrainingConfig.MODEL_NAME,
        "trainable_parameters": trainable_params,
        "all_parameters": all_param,
        "trainable_percent": 100 * trainable_params / all_param,
        "validation_loss": val_loss,
        "perplexity": perplexity,
        "pre_training_generation": pre_gen,
        "post_training_generation": post_gen,
        "status": "completed"
    }
    
    report_file = Path("eval_report.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
    logger.info("Evaluation report generated at eval_report.json")

if __name__ == "__main__":
    run_training()
