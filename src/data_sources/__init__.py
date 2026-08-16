"""
SecureLoRA Data Sources Adapter Package
=======================================
Provides a unified dataset adapter layer for:
  - AI4Privacy PII-Masking dataset (HF: ai4privacy/pii-masking-300k)
  - Synthea / SyntheticMass healthcare records
  - SecureLoRA synthetic benchmark dataset
"""

from src.data_sources.base_dataset import BaseDatasetAdapter
from src.data_sources.ai4privacy_loader import AI4PrivacyDatasetAdapter
from src.data_sources.synthea_loader import SyntheaDatasetAdapter
from src.data_sources.synthetic_loader import SyntheticDatasetAdapter
from src.data_sources.dataset_registry import dataset_registry, get_dataset_adapter

__all__ = [
    "BaseDatasetAdapter",
    "AI4PrivacyDatasetAdapter",
    "SyntheaDatasetAdapter",
    "SyntheticDatasetAdapter",
    "dataset_registry",
    "get_dataset_adapter",
]
