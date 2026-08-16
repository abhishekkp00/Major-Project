"""
synthetic_loader.py
===================
Dataset adapter for the existing SecureLoRA synthetic benchmark.
Features:
  - Loads existing benchmark datasets (sample_pii_data.jsonl, sample_medical_phi.jsonl, real_world_pii.jsonl).
  - Built-in generator for expandable synthetic benchmark records with ground-truth entity spans.
  - Useful for controlled testing, ground-truth PII spans, repeatable unit tests, and adversarial experiments.
  - Sets `ground_truth_available = True` and `synthetic_source = True`.
  - Records generator version and seed for exact reproducibility.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from src.data_sources.base_dataset import BaseDatasetAdapter

logger = logging.getLogger("secure_lora.data_sources.synthetic")


class SyntheticDatasetAdapter(BaseDatasetAdapter):
    """Adapter for existing SecureLoRA synthetic PII/PHI benchmark."""

    DEFAULT_SUBSET_SIZE = 10000

    def __init__(self):
        super().__init__(
            dataset_id="synthetic",
            dataset_name="SecureLoRA Synthetic PII/PHI Benchmark",
            source="SecureLoRA Synthetic Benchmark Generator",
            license_info="MIT License / Project Internal Benchmark",
            attribution="SecureLoRA Framework Controlled Synthetic Benchmark",
            synthetic_source=True,
            ground_truth_available=True,
            domain="Controlled Enterprise & Healthcare PII/PHI",
            version="1.5.0",
            redistribution_permitted=True
        )
        self.generator_version = "1.5.0"
        self._revision = f"v{self.generator_version}"

    def load_dataset(
        self,
        subset_size: Optional[int] = None,
        split: str = "train",
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Loads SecureLoRA synthetic benchmark records."""
        eff_subset_size = subset_size if subset_size is not None else self.DEFAULT_SUBSET_SIZE
        self._current_split = split
        self._download_timestamp = datetime.now(timezone.utc).isoformat()

        raw_records = []

        # 1. Search project root & samples for JSONL files
        sample_files = [
            Path("real_world_pii.jsonl"),
            Path("samples/sample_pii_data.jsonl"),
            Path("samples/sample_medical_phi.jsonl"),
            Path("sample_pii_data.jsonl"),
            Path("sample_medical_phi.jsonl")
        ]

        for sfile in sample_files:
            if sfile.exists():
                try:
                    with open(sfile, "r", encoding="utf-8") as fp:
                        for line_num, line in enumerate(fp, 1):
                            if line.strip():
                                obj = json.loads(line)
                                obj["source_file"] = sfile.name
                                raw_records.append(obj)
                except Exception as e:
                    logger.warning("Failed reading sample file %s: %s", sfile, e)

        # 2. If raw records count is less than subset_size, expand using synthetic generator
        if len(raw_records) < eff_subset_size:
            needed = eff_subset_size - len(raw_records)
            logger.info("Expanding synthetic benchmark with %d generated records (seed=%d)...", needed, seed)
            generated = self._generate_synthetic_benchmark_records(needed, seed=seed)
            raw_records.extend(generated)

        # Sample deterministically
        sampled_raw = self._sample_records(raw_records, eff_subset_size, seed=seed)
        self._effective_subset_size = len(sampled_raw)

        # Normalize records
        normalized = []
        for idx, rec in enumerate(sampled_raw, 1):
            rec_id = rec.get("record_id") or rec.get("id") or f"sec_syn_{split}_{idx:05d}"
            instruction = rec.get("instruction") or rec.get("prompt") or "Redact Personally Identifiable Information (PII) from this text."
            input_text = rec.get("input") or rec.get("text") or ""
            output_text = rec.get("output") or rec.get("masked_text") or ""

            # Extract or compute ground truth entities
            pii_entities = rec.get("pii_entities")
            if pii_entities is None:
                pii_entities = self._extract_ground_truth_entities(input_text)

            norm_rec = self.normalize_record(
                record_id=rec_id,
                domain=rec.get("domain", self.domain),
                instruction=instruction,
                input_text=input_text,
                output_text=output_text,
                source_dataset="SecureLoRA Synthetic Benchmark",
                synthetic_source=True,
                pii_entities=pii_entities
            )
            norm_rec["generator_version"] = self.generator_version
            norm_rec["seed"] = seed
            norm_rec["source_split"] = split
            normalized.append(norm_rec)

        self._loaded_records = normalized
        return self._loaded_records

    def _extract_ground_truth_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extracts exact regex-based ground truth PII spans for synthetic benchmark records."""
        from src.evaluation.pii_metrics import _PII_PATTERNS
        entities = []
        for ptype, pat in _PII_PATTERNS.items():
            for m in pat.finditer(text):
                entities.append({
                    "type": ptype,
                    "start": m.start(),
                    "end": m.end(),
                    "text": m.group(0)
                })
        return entities

    def _generate_synthetic_benchmark_records(self, count: int, seed: int = 42) -> List[Dict[str, Any]]:
        """Generates deterministic synthetic benchmark records with ground-truth entity spans."""
        import random
        rng = random.Random(seed)

        templates = [
            ("Employee {name}, SSN: {ssn}, email {email}, phone {phone}.",
             "Employee [NAME], SSN: [MASKED_SSN], email [MASKED_EMAIL], phone [MASKED_PHONE]."),
            ("Patient {name} (DOB: {dob}) admitted with IP {ip}. Emergency contact: {phone}.",
             "Patient [NAME] (DOB: [DATE]) admitted with IP [MASKED_IP]. Emergency contact: [MASKED_PHONE]."),
            ("Billing inquiry from {name}, account email {email}, card number {card}.",
             "Billing inquiry from [NAME], account email [MASKED_EMAIL], card number [MASKED_CARD]."),
            ("System log: user {email} failed login from {ip} with secret api_key='{key}'.",
             "System log: user [MASKED_EMAIL] failed login from [MASKED_IP] with secret api_key='[MASKED_SECRET]'.")
        ]

        names = ["Avery Miller", "Blake Taylor", "Cameron Davis", "Drew Wilson", "Emerson Clark"]
        emails = ["avery.m@corp.org", "blake.t@company.net", "cameron.d@tech.io", "drew.w@admin.com"]
        phones = ["312-555-0144", "415-555-0177", "212-555-0199", "650-555-0133"]
        ssns = ["333-22-1111", "555-44-3333", "777-88-9999", "111-99-8888"]
        ips = ["10.0.1.45", "192.168.2.100", "172.16.5.20", "203.0.113.88"]
        cards = ["4111-1111-1111-1111", "5500-0000-0000-0004", "3714-496353-98431"]
        keys = ["sk_live_998877665544332211", "sec_key_a1b2c3d4e5f67890", "tok_xyz987654321"]
        dobs = ["1985-04-12", "1992-09-25", "1978-11-03", "2001-01-18"]

        records = []
        for i in range(count):
            tmpl, target_tmpl = rng.choice(templates)
            n = rng.choice(names)
            e = rng.choice(emails)
            p = rng.choice(phones)
            s = rng.choice(ssns)
            ip_val = rng.choice(ips)
            c_val = rng.choice(cards)
            k_val = rng.choice(keys)
            d_val = rng.choice(dobs)

            inp = tmpl.format(name=n, ssn=s, email=e, phone=p, dob=d_val, ip=ip_val, card=c_val, key=k_val)
            out = target_tmpl

            records.append({
                "record_id": f"gen_syn_{i+1:06d}",
                "domain": "Controlled PII/PHI Synthetic Benchmark",
                "instruction": "Redact Personally Identifiable Information (PII) from this text.",
                "input": inp,
                "output": out,
                "source_dataset": "SecureLoRA Synthetic Benchmark",
                "synthetic_source": True
            })
        return records
