import os
import json
import subprocess
from pathlib import Path
from datasets import load_dataset

def main():
    print("Step 1: Fetching Hugging Face dataset 'ai4privacy/openpii-masking-mini-10k'...")
    try:
        # Load train split of dataset
        dataset = load_dataset("ai4privacy/openpii-masking-mini-10k", split="train")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return

    # Ingest 150 samples for fast, secure local CPU training validation
    subset_size = 150
    records = []
    
    print(f"Step 2: Processing first {subset_size} samples into standard instruction format...")
    for idx in range(min(subset_size, len(dataset))):
        sample = dataset[idx]
        records.append({
            "instruction": "Mask all Personally Identifiable Information (PII) in the text.",
            "input": sample["source_text"],
            "output": sample["masked_text"]
        })
        
    input_dir = Path("real_data_inputs")
    # Clean previous files to avoid mixing datasets
    if input_dir.exists():
        for f in input_dir.glob("*"):
            if f.is_file():
                f.unlink()
    input_dir.mkdir(exist_ok=True)
    
    # Save the normalized JSON dataset
    dest_path = input_dir / "openpii_masking.json"
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
        
    print(f"Step 3: Standardized dataset saved to {dest_path}")
    
    # Run the GCM encryption tool
    print("Step 4: Encrypting dataset via CLI pipeline...")
    enc_cmd = [
        "python3", "-m", "secure_lora.cli", "encrypt",
        "-i", "real_data_inputs",
        "-o", "encrypted_real_data",
        "-n", "OpenPIIMaskingDataset"
    ]
    
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Execute encryption
    res = subprocess.run(enc_cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print("Encryption failed!")
        print(res.stderr)
        return
        
    print(res.stdout)
    print("Step 5: Dataset successfully encrypted. Ready for secure model training.")

if __name__ == "__main__":
    main()
