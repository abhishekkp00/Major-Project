"""
dataset_registry.py
===================
Canonical Dataset Registry for SecureLoRA.
Registers and provides access to all dataset adapters:
  1. AI4Privacy PII-Masking Benchmark (ai4privacy)
  2. Synthea / SyntheticMass Healthcare Records (synthea)
  3. SecureLoRA Synthetic Benchmark (synthetic)
"""

import logging
from typing import Dict, Any, List, Optional
from src.data_sources.base_dataset import BaseDatasetAdapter
from src.data_sources.ai4privacy_loader import AI4PrivacyDatasetAdapter
from src.data_sources.synthea_loader import SyntheaDatasetAdapter
from src.data_sources.synthetic_loader import SyntheticDatasetAdapter

logger = logging.getLogger("secure_lora.data_sources.registry")


class DatasetRegistry:
    """Registry managing dataset adapters for SecureLoRA."""

    def __init__(self):
        self._adapters: Dict[str, BaseDatasetAdapter] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Registers the three core dataset adapters."""
        self.register_adapter(AI4PrivacyDatasetAdapter())
        self.register_adapter(SyntheaDatasetAdapter())
        self.register_adapter(SyntheticDatasetAdapter())

    def register_adapter(self, adapter: BaseDatasetAdapter):
        """Registers a dataset adapter instance."""
        if not isinstance(adapter, BaseDatasetAdapter):
            raise TypeError(f"Adapter must be an instance of BaseDatasetAdapter, got {type(adapter)}")
        self._adapters[adapter.dataset_id.lower()] = adapter
        logger.info("Registered dataset adapter: %s (%s)", adapter.dataset_id, adapter.dataset_name)

    def get_dataset_adapter(self, dataset_id: str) -> BaseDatasetAdapter:
        """Retrieves an adapter by ID (case-insensitive)."""
        key = dataset_id.lower().strip()
        # Aliases / mapping for backward compatibility
        alias_map = {
            "ai4p": "ai4privacy",
            "ai4privacy": "ai4privacy",
            "ai4privacy/pii-masking-300k": "ai4privacy",
            "pii_corporate": "ai4privacy",
            "real_world_pii": "ai4privacy",
            "synthea": "synthea",
            "clinical_notes": "synthea",
            "synthetic": "synthetic",
            "sample_pii_data.jsonl": "synthetic",
            "sample_medical_phi.jsonl": "synthea"
        }

        canonical_key = alias_map.get(key, key)
        if canonical_key not in self._adapters:
            available = list(self._adapters.keys())
            raise KeyError(f"Dataset adapter '{dataset_id}' not found. Registered datasets: {available}")
        return self._adapters[canonical_key]

    def list_datasets(self) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all registered dataset adapters."""
        dataset_list = []
        for adapter in self._adapters.values():
            meta = adapter.get_metadata()
            dataset_list.append({
                "id": adapter.dataset_id,
                "name": adapter.dataset_name,
                "source": adapter.source,
                "license": adapter.license_info,
                "attribution": adapter.attribution,
                "domain": adapter.domain,
                "ground_truth_available": adapter.ground_truth_available,
                "synthetic_source": adapter.synthetic_source,
                "redistribution_permitted": adapter.redistribution_permitted,
                "default_subset_size": getattr(adapter, "DEFAULT_SUBSET_SIZE", 100),
                "subset_options": [50, 100, 500, 1000, 5000, 10000],
                "version": adapter.version,
                "revision": meta.get("revision", "main")
            })
        return dataset_list


    def load_dataset(
        self,
        dataset_id: str,
        subset_size: Optional[int] = None,
        split: str = "train",
        seed: int = 42
    ) -> BaseDatasetAdapter:
        """Helper to retrieve adapter and load dataset records in one call."""
        adapter = self.get_dataset_adapter(dataset_id)
        adapter.load_dataset(subset_size=subset_size, split=split, seed=seed)
        return adapter


# Global Singleton Registry Instance
dataset_registry = DatasetRegistry()


def get_dataset_adapter(dataset_id: str) -> BaseDatasetAdapter:
    """Utility function to get adapter from default registry."""
    return dataset_registry.get_dataset_adapter(dataset_id)
