"""
base_dataset.py
===============
Base abstract interface for SecureLoRA dataset adapters.
All dataset adapters (AI4Privacy, Synthea, Synthetic) inherit from BaseDatasetAdapter
and normalize records to SecureLoRA's canonical JSONL schema.
"""

import abc
import random
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone


class BaseDatasetAdapter(abc.ABC):
    """Abstract Base Class for SecureLoRA Dataset Adapters."""

    def __init__(
        self,
        dataset_id: str,
        dataset_name: str,
        source: str,
        license_info: str,
        attribution: str,
        synthetic_source: Union[bool, str],
        ground_truth_available: bool,
        domain: str,
        version: str = "1.0.0",
        redistribution_permitted: bool = False
    ):
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.source = source
        self.license_info = license_info
        self.attribution = attribution
        self.synthetic_source = synthetic_source
        self.ground_truth_available = ground_truth_available
        self.domain = domain
        self.version = version
        self.redistribution_permitted = redistribution_permitted

        self._loaded_records: List[Dict[str, Any]] = []
        self._effective_subset_size: int = 0
        self._current_split: str = "train"
        self._download_timestamp: Optional[str] = None
        self._revision: str = "main"

    @abc.abstractmethod
    def load_dataset(
        self,
        subset_size: Optional[int] = None,
        split: str = "train",
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Loads dataset records, subsamples deterministically with seed if subset_size specified,
        normalizes every record to SecureLoRA schema, and caches in self._loaded_records.
        """
        pass

    def normalize_record(
        self,
        record_id: str,
        domain: str,
        instruction: str,
        input_text: str,
        output_text: str,
        source_dataset: str,
        synthetic_source: Union[bool, str],
        pii_entities: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Converts any source record into SecureLoRA's internal normalized schema:
        {
            "record_id": "...",
            "domain": "...",
            "instruction": "...",
            "input": "...",
            "output": "...",
            "source_dataset": "...",
            "synthetic_source": true/false or "Synthea",
            "pii_entities": [...]
        }
        """
        return {
            "record_id": str(record_id),
            "domain": str(domain),
            "instruction": str(instruction or "").strip(),
            "input": str(input_text or "").strip(),
            "output": str(output_text or "").strip(),
            "source_dataset": str(source_dataset),
            "synthetic_source": synthetic_source,
            "pii_entities": pii_entities if pii_entities is not None else []
        }

    def get_metadata(self) -> Dict[str, Any]:
        """Returns structured metadata about the dataset adapter state."""
        return {
            "name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "dataset_name": self.dataset_name,
            "version": self.version,
            "revision": self._revision,
            "source": self.source,
            "license": self.license_info,
            "attribution": self.attribution,
            "record_count": len(self._loaded_records),
            "domain": self.domain,
            "ground_truth_available": self.ground_truth_available,
            "synthetic_source": self.synthetic_source,
            "redistribution_permitted": self.redistribution_permitted,
            "download_timestamp": self._download_timestamp or datetime.now(timezone.utc).isoformat(),
            "split_information": self.get_split_information(),
            "split": self._current_split,
            "subset_size": self._effective_subset_size,
            "total_loaded_records": len(self._loaded_records)
        }

    def load(
        self,
        subset_size: Optional[int] = None,
        split: str = "train",
        seed: int = 42,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Common interface method: loads dataset records with optional subset_size."""
        return self.load_dataset(subset_size=subset_size, split=split, seed=seed, **kwargs)

    def metadata(self) -> Dict[str, Any]:
        """Common interface method: returns dataset metadata."""
        return self.get_metadata()

    def records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Common interface method: returns loaded records."""
        return self.get_records(limit=limit)

    def split(
        self,
        train_ratio: float = 0.8,
        seed: int = 42,
        split_name: Optional[str] = None
    ) -> Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Common interface method: returns dataset splits or train/test dict."""
        if split_name is not None:
            return self.get_split(split_name=split_name)
        if not self._loaded_records:
            return {"train": [], "test": []}
        rng = random.Random(seed)
        shuffled = list(self._loaded_records)
        rng.shuffle(shuffled)
        split_idx = int(len(shuffled) * train_ratio)
        return {
            "train": shuffled[:split_idx],
            "test": shuffled[split_idx:]
        }

    def ground_truth(self) -> Dict[str, Any]:
        """Common interface method: returns ground truth PII information."""
        return self.get_ground_truth()

    def statistics(self) -> Dict[str, Any]:
        """Common interface method: returns dataset statistics."""
        return self.get_statistics()

    def get_split_information(self) -> Dict[str, Any]:
        """Exposes split information details."""
        return {
            "current_split": self._current_split,
            "subset_size": self._effective_subset_size,
            "total_loaded_records": len(self._loaded_records)
        }

    def get_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns the loaded records (up to limit if specified)."""
        if limit is not None and limit > 0:
            return self._loaded_records[:limit]
        return self._loaded_records

    def get_split(self, split_name: str = "train") -> List[Dict[str, Any]]:
        """Filter loaded records by split if split attribute present, or load split."""
        split_records = [
            r for r in self._loaded_records
            if r.get("source_split") == split_name or r.get("split") == split_name
        ]
        if not split_records and self._loaded_records:
            return self._loaded_records
        return split_records

    def get_ground_truth(self) -> Dict[str, Any]:
        """
        Returns ground truth PII information for evaluation.
        If ground truth is unavailable: ground_truth_available = false.
        """
        if not self.ground_truth_available:
            return {
                "ground_truth_available": False,
                "reason": f"Ground truth entity annotations unavailable for dataset {self.dataset_id}",
                "total_records": len(self._loaded_records),
                "total_annotated_entities": 0,
                "pii_entities_by_type": {}
            }

        entity_counts: Dict[str, int] = {}
        total_entities = 0
        records_with_entities = 0

        for r in self._loaded_records:
            entities = r.get("pii_entities", [])
            if entities:
                records_with_entities += 1
                total_entities += len(entities)
                for ent in entities:
                    ent_type = ent.get("type", "UNKNOWN")
                    entity_counts[ent_type] = entity_counts.get(ent_type, 0) + 1

        return {
            "ground_truth_available": True,
            "total_records": len(self._loaded_records),
            "records_with_pii": records_with_entities,
            "total_annotated_entities": total_entities,
            "pii_entities_by_type": entity_counts
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Returns dataset statistics (record count, average lengths, PII summary)."""
        num_records = len(self._loaded_records)
        if num_records == 0:
            return {
                "num_records": 0,
                "avg_input_len": 0,
                "avg_output_len": 0,
                "ground_truth_available": self.ground_truth_available
            }

        total_input_chars = sum(len(r.get("input", "")) for r in self._loaded_records)
        total_output_chars = sum(len(r.get("output", "")) for r in self._loaded_records)

        stats = {
            "num_records": num_records,
            "avg_input_len": round(total_input_chars / num_records, 2),
            "avg_output_len": round(total_output_chars / num_records, 2),
            "ground_truth_available": self.ground_truth_available,
            "domain": self.domain,
            "synthetic_source": self.synthetic_source,
            "license": self.license_info
        }

        gt = self.get_ground_truth()
        if gt.get("ground_truth_available"):
            stats["total_annotated_entities"] = gt.get("total_annotated_entities", 0)
            stats["pii_entity_types"] = gt.get("pii_entities_by_type", {})
        return stats

    def _sample_records(
        self,
        records: List[Dict[str, Any]],
        subset_size: Optional[int],
        seed: int = 42
    ) -> List[Dict[str, Any]]:
        """Utility method to perform reproducible subsampling."""
        if subset_size is None or subset_size >= len(records) or subset_size <= 0:
            return records
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(records)), subset_size)
        sampled_indices.sort()
        return [records[i] for i in sampled_indices]
