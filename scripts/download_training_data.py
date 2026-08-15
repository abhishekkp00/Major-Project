"""
download_training_data.py

Downloads 500 high-quality, real-world PII masking examples from:
  ai4privacy/pii-masking-300k  (HuggingFace)

Each record contains:
  - source_text : original sentence with real PII
  - target_text : same sentence with PII replaced by typed tokens

The output is saved as:
  real_data_inputs/pii_training_data.jsonl  (used for dashboard training)
  real_world_pii.jsonl                      (shown in dashboard demo panel)
"""

import json
import sys
from pathlib import Path
from datasets import load_dataset

TARGET_RECORDS = 500
DEMO_RECORDS   = 50   # shown in the dashboard demo panel
OUTPUT_TRAIN   = Path("real_data_inputs/pii_training_data.jsonl")
OUTPUT_DEMO    = Path("real_world_pii.jsonl")

# PII token normalisation map — maps ai4privacy tags → our project tokens
TOKEN_MAP = {
    "GIVENNAME1":    "[GIVENNAME]",
    "GIVENNAME2":    "[GIVENNAME]",
    "SURNAME1":      "[SURNAME]",
    "SURNAME2":      "[SURNAME]",
    "EMAIL":         "[EMAIL]",
    "TELEPHONENUM":  "[TEL]",
    "SOCIALNUMBER":  "[SOCIALNUMBER]",
    "IDNUM":         "[IDNUM]",
    "DRIVERLICENSE": "[DRIVERLICENSE]",
    "PASSPORT":      "[PASSPORT]",
    "STREET":        "[STREET]",
    "CITY":          "[CITY]",
    "STATE":         "[STATE]",
    "ZIPCODE":       "[ZIPCODE]",
    "COUNTRY":       "[COUNTRY]",
    "DATE":          "[DATE]",
    "TIME":          "[TIME]",
    "AGE":           "[AGE]",
    "USERNAME":      "[USERNAME]",
    "PASSWORD":      "[PASSWORD]",
    "CREDITCARDNUMBER": "[CREDITCARD]",
    "ACCOUNTNUMBER": "[ACCOUNTNUM]",
    "IBAN":          "[IBAN]",
    "IPV4":          "[IPADDRESS]",
    "IPV6":          "[IPADDRESS]",
    "URL":           "[URL]",
    "COMPANY":       "[COMPANY]",
}

def normalize_target(target_text: str) -> str:
    """Replace ai4privacy token format with our project token format."""
    result = target_text
    for src, dst in TOKEN_MAP.items():
        result = result.replace(f"[{src}]", dst)
    return result

def main():
    print("Downloading ai4privacy/pii-masking-300k from HuggingFace...")
    print("(This is a real-world, pre-labeled PII dataset with 300,000 examples)")

    try:
        ds = load_dataset(
            "ai4privacy/pii-masking-300k",
            split="train",
            streaming=True,        # Streaming — no need to download all 300k
            trust_remote_code=True
        )
    except Exception as e:
        print(f"ERROR: Could not load dataset — {e}", file=sys.stderr)
        sys.exit(1)

    records = []
    seen    = set()

    for row in ds:
        if len(records) >= TARGET_RECORDS:
            break

        source = (row.get("source_text") or "").replace("\n", " ").replace("\r", " ").strip()
        target = (row.get("target_text") or "").replace("\n", " ").replace("\r", " ").strip()
        mask   = row.get("privacy_mask") or []

        # Quality filters:
        # - Must have at least 1 PII entity
        # - Source and target must differ (i.e., something was actually masked)
        # - Reasonable length for a 68M model (50-350 chars)
        if not mask:
            continue
        if source == target:
            continue
        if not (50 < len(source) <= 350):
            continue
        if source in seen:
            continue

        seen.add(source)
        target_normalized = normalize_target(target)

        records.append({
            "instruction": f"Redact Personally Identifiable Information (PII) from this text: {source}",
            "output":      f"Redact Personally Identifiable Information (PII) from this text: {target_normalized}"
        })

        if len(records) % 50 == 0:
            print(f"  Collected {len(records)} / {TARGET_RECORDS} records...")

    if not records:
        print("ERROR: No records collected. Check your internet connection.", file=sys.stderr)
        sys.exit(1)

    print(f"\nCollected {len(records)} real-world PII records.")

    # Save full training set
    OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} training records → {OUTPUT_TRAIN}")

    # Save demo subset (first 50 diverse records)
    with open(OUTPUT_DEMO, "w", encoding="utf-8") as f:
        for rec in records[:DEMO_RECORDS]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {DEMO_RECORDS} demo records → {OUTPUT_DEMO}")

    # Print a few examples so the user can verify
    print("\n--- Sample Records ---")
    for rec in records[:3]:
        src = rec["instruction"].replace("Redact Personally Identifiable Information (PII) from this text: ", "")
        tgt = rec["output"].replace("Redact Personally Identifiable Information (PII) from this text: ", "")
        print(f"  INPUT:  {src[:120]}")
        print(f"  OUTPUT: {tgt[:120]}")
        print()

if __name__ == "__main__":
    main()
