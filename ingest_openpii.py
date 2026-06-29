import os
import json
import argparse
import subprocess
from pathlib import Path
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser(description="Fetch and encrypt the OpenPII dataset for secure training.")
    parser.add_argument("-l", "--limit", type=int, default=150, help="Max samples to import (default: 150)")
    parser.add_argument("-d", "--dataset", type=str, default="ai4privacy/openpii-masking-mini-10k", help="Hugging Face dataset path")
    args = parser.parse_args()

    print(f"Downloading {args.dataset} from Hugging Face...")
    try:
        dataset = load_dataset(args.dataset, split="train")
    except Exception as e:
        print(f"Failed to fetch dataset: {e}")
        return

    records = []
    limit = min(args.limit, len(dataset))
    print(f"Formatting first {limit} records for instruction-tuning...")
    
    for idx in range(limit):
        item = dataset[idx]
        records.append({
            "instruction": "Mask all Personally Identifiable Information (PII) in the text.",
            "input": item["source_text"],
            "output": item["masked_text"]
        })
        
    input_dir = Path("real_data_inputs")
    if input_dir.exists():
        for file_path in input_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
    input_dir.mkdir(exist_ok=True)
    
    output_json = input_dir / "openpii_masking.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=4)
        
    print(f"Saved preprocessed records to {output_json}")
    
    print("Running encryption command...")
    cmd = [
        "python3", "-m", "secure_lora.cli", "encrypt",
        "-i", "real_data_inputs",
        "-o", "encrypted_real_data",
        "-n", "OpenPIIMaskingDataset"
    ]
    
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = "."
    
    res = subprocess.run(cmd, env=run_env, capture_output=True, text=True)
    if res.returncode != 0:
        print("Encryption failed:")
        print(res.stderr)
        return
        
    print(res.stdout)
    print("Encryption finished successfully.")

if __name__ == "__main__":
    main()
