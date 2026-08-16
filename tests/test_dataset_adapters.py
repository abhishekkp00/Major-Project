"""
test_dataset_adapters.py
========================
Comprehensive test suite verifying the SecureLoRA Dataset Adapter Layer (STEP 2):
  - BaseDatasetAdapter common interface (load, metadata, records, split, ground_truth, statistics)
  - AI4Privacy adapter (ground-truth spans, HF integration, fallback generator, configurable subsets)
  - Synthea adapter (EHR narratives, structural metadata, ground_truth_available = False)
  - Synthetic adapter (legacy benchmarks & exact span generator, ground_truth_available = True)
  - DatasetRegistry lookup, alias resolution, and invalid/missing dataset handling
  - Mandatory Metadata Schema compliance (name, version/revision, source, license, record_count, domain, ground_truth_available, split_information)
"""

import os
import unittest
from pathlib import Path

from src.data_sources.dataset_registry import dataset_registry, get_dataset_adapter
from src.data_sources.base_dataset import BaseDatasetAdapter
from src.data_sources.ai4privacy_loader import AI4PrivacyDatasetAdapter
from src.data_sources.synthea_loader import SyntheaDatasetAdapter
from src.data_sources.synthetic_loader import SyntheticDatasetAdapter
from src.orchestrator.service import JobOrchestrator


class TestDatasetAdapterLayer(unittest.TestCase):

    def setUp(self):
        os.environ["HF_HUB_OFFLINE"] = "1"

    def test_registry_registration_and_aliases(self):
        """Tests that all adapters are properly registered and aliases work."""
        registered = dataset_registry.list_datasets()
        registered_ids = [d["id"] for d in registered]

        self.assertIn("ai4privacy", registered_ids)
        self.assertIn("synthea", registered_ids)
        self.assertIn("synthetic", registered_ids)

        # Check aliases
        ai4p_1 = dataset_registry.get_dataset_adapter("ai4p")
        ai4p_2 = dataset_registry.get_dataset_adapter("ai4privacy/pii-masking-300k")
        self.assertIsInstance(ai4p_1, AI4PrivacyDatasetAdapter)
        self.assertIsInstance(ai4p_2, AI4PrivacyDatasetAdapter)

        syn = dataset_registry.get_dataset_adapter("synthea")
        self.assertIsInstance(syn, SyntheaDatasetAdapter)

        sec_syn = dataset_registry.get_dataset_adapter("synthetic")
        self.assertIsInstance(sec_syn, SyntheticDatasetAdapter)

    def test_invalid_and_missing_dataset(self):
        """Tests lookup failure for non-existent or invalid dataset IDs."""
        with self.assertRaises(KeyError):
            dataset_registry.get_dataset_adapter("non_existent_dataset_xyz")

        with self.assertRaises(KeyError):
            dataset_registry.get_dataset_adapter("")

    def test_common_interface_methods(self):
        """Tests common interface methods: load(), metadata(), records(), split(), ground_truth(), statistics()."""
        adapters = [
            AI4PrivacyDatasetAdapter(),
            SyntheaDatasetAdapter(),
            SyntheticDatasetAdapter()
        ]

        for adapter in adapters:
            # 1. load()
            recs = adapter.load(subset_size=25, seed=42)
            self.assertEqual(len(recs), 25)

            # 2. metadata()
            meta = adapter.metadata()
            self.assertIsInstance(meta, dict)
            self.assertIn("name", meta)
            self.assertIn("version", meta)
            self.assertIn("revision", meta)
            self.assertIn("source", meta)
            self.assertIn("license", meta)
            self.assertIn("record_count", meta)
            self.assertEqual(meta["record_count"], 25)
            self.assertIn("domain", meta)
            self.assertIn("ground_truth_available", meta)
            self.assertIn("split_information", meta)

            # 3. records()
            loaded_recs = adapter.records(limit=10)
            self.assertEqual(len(loaded_recs), 10)

            # 4. split()
            splits = adapter.split(train_ratio=0.8, seed=42)
            self.assertIn("train", splits)
            self.assertIn("test", splits)
            self.assertEqual(len(splits["train"]) + len(splits["test"]), 25)

            # 5. ground_truth()
            gt = adapter.ground_truth()
            self.assertIn("ground_truth_available", gt)
            self.assertEqual(gt["ground_truth_available"], adapter.ground_truth_available)

            # 6. statistics()
            stats = adapter.statistics()
            self.assertEqual(stats["num_records"], 25)
            self.assertIn("avg_input_len", stats)
            self.assertIn("avg_output_len", stats)

    def test_dataset_metadata_schema(self):
        """Validates that every dataset adapter exposes all required metadata fields."""
        for ds_info in dataset_registry.list_datasets():
            adapter = dataset_registry.get_dataset_adapter(ds_info["id"])
            adapter.load(subset_size=15, seed=42)
            meta = adapter.metadata()

            required_fields = [
                "name", "version", "revision", "source", "license",
                "record_count", "domain", "ground_truth_available", "split_information"
            ]

            for field in required_fields:
                self.assertIn(field, meta, f"Field '{field}' missing from metadata for dataset '{ds_info['id']}'")

    def test_ai4privacy_normalization_and_sampling(self):
        """Tests AI4Privacy adapter record normalization, configurable subsets, and reproducible sampling."""
        adapter = AI4PrivacyDatasetAdapter()
        recs = adapter.load(subset_size=50, seed=42, force_offline=True)

        self.assertEqual(len(recs), 50)
        first = recs[0]

        # Verify canonical schema fields
        expected_keys = {"record_id", "domain", "instruction", "input", "output", "source_dataset", "synthetic_source", "pii_entities"}
        self.assertTrue(expected_keys.issubset(first.keys()))
        self.assertEqual(first["source_dataset"], "ai4privacy/pii-masking-300k")

        # Test metadata
        meta = adapter.metadata()
        self.assertEqual(meta["dataset_id"], "ai4privacy")
        self.assertTrue(meta["ground_truth_available"])
        self.assertFalse(meta["redistribution_permitted"])

        # Test seed reproducibility
        recs_seed42_a = adapter.load(subset_size=10, seed=42, force_offline=True)
        recs_seed42_b = adapter.load(subset_size=10, seed=42, force_offline=True)
        self.assertEqual([r["record_id"] for r in recs_seed42_a], [r["record_id"] for r in recs_seed42_b])

    def test_synthea_loader(self):
        """Tests Synthea clinical note adapter and ground_truth_available = False enforcement."""
        adapter = SyntheaDatasetAdapter()
        recs = adapter.load(subset_size=30, seed=42)

        self.assertEqual(len(recs), 30)
        first = recs[0]

        self.assertTrue(first["record_id"].lower().startswith("synth"))
        self.assertIn("Clinical Records", first["domain"])
        self.assertIn("clinical_metadata", first)
        self.assertIn("conditions", first["clinical_metadata"])
        self.assertIn("medications", first["clinical_metadata"])

        meta = adapter.metadata()
        self.assertFalse(meta["ground_truth_available"])
        self.assertTrue(meta["redistribution_permitted"])

        gt = adapter.ground_truth()
        self.assertFalse(gt["ground_truth_available"])
        self.assertEqual(gt["total_annotated_entities"], 0)

        stats = adapter.statistics()
        self.assertEqual(stats["num_records"], 30)

    def test_synthetic_loader(self):
        """Tests Synthetic benchmark adapter."""
        adapter = SyntheticDatasetAdapter()
        recs = adapter.load(subset_size=20, seed=42)

        self.assertEqual(len(recs), 20)
        first = recs[0]

        self.assertEqual(first["source_dataset"], "SecureLoRA Synthetic Benchmark")
        self.assertTrue(first["synthetic_source"])

        meta = adapter.metadata()
        self.assertTrue(meta["ground_truth_available"])

    def test_orchestrator_integration(self):
        """Tests JobOrchestrator dataset adapter job creation."""
        tmp_jobs_dir = Path("outputs/test_jobs")
        tmp_jobs_dir.mkdir(parents=True, exist_ok=True)
        orchestrator_inst = JobOrchestrator(base_jobs_dir=str(tmp_jobs_dir))

        job_id = orchestrator_inst.create_job(
            dataset_name="AI4Privacy PII Benchmark",
            dataset_type="ai4privacy",
            subset_size=20,
            epochs=1
        )
        job = orchestrator_inst.get_job(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job["dataset_type"], "ai4privacy")
        self.assertEqual(job["subset_size"], 20)


if __name__ == "__main__":
    unittest.main()
