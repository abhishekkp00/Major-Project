import urllib.request
import json
import os
import sys

def main():
    print("Fetching real-world PII dataset from Hugging Face...")
    # Fetch 150 rows to find diverse and interesting samples
    url = 'https://datasets-server.huggingface.co/rows?dataset=ai4privacy/pii-masking-300k&config=default&split=train&offset=0&limit=150'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"Error calling Hugging Face API: {e}", file=sys.stderr)
        sys.exit(1)

    rows = []
    seen = set()
    for row_info in data.get('rows', []):
        r = row_info.get('row', {})
        source = r.get('source_text', '').strip()
        target = r.get('target_text', '').strip()
        mask = r.get('privacy_mask', [])
        
        # Select records with PII, under 400 characters, and unique content
        if source != target and 50 < len(source) < 400 and mask:
            # Clean up newlines or weird formatting to make it clean JSONL
            source_clean = source.replace("\n", " ").replace("\r", " ")
            target_clean = target.replace("\n", " ").replace("\r", " ")
            if source_clean not in seen:
                seen.add(source_clean)
                rows.append({
                    "instruction": f"Redact Personally Identifiable Information (PII) from this text: {source_clean}",
                    "output": f"Redact Personally Identifiable Information (PII) from this text: {target_clean}"
                })

    print(f"Filtered {len(rows)} high-quality real-world PII records.")
    
    # We will write the first 10 rows for optimal CPU training speed in the demo
    output_path = "real_world_pii.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows[:10]:
            f.write(json.dumps(row) + "\n")
            
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    main()
