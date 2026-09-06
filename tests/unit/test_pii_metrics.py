"""
test_pii_metrics.py
===================
Unit test for PII metrics evaluation path.
"""

import unittest
from src.evaluation.pii_metrics import evaluate_pii_detection, GROUND_TRUTH_CORPUS, EVAL_PII_CLASSES


class TestPIIMetrics(unittest.TestCase):

    def test_evaluate_pii_detection_structure(self):
        """Tests that evaluate_pii_detection returns required metrics, metadata, and provenance."""
        res = evaluate_pii_detection(verbose=False)

        # 1. Provenance check
        self.assertIn("detector_provenance", res)
        self.assertIn("HybridPIIEngine", res["detector_provenance"])

        # 2. Metadata check
        meta = res["metadata"]
        self.assertEqual(meta["corpus_size"], len(GROUND_TRUTH_CORPUS))
        self.assertIn("HybridPIIEngine", meta["detector_provenance"])
        self.assertIn("scope_disclaimer", meta)
        self.assertIn("Does NOT measure live LLM memorization", meta["scope_disclaimer"])

        # 3. Per-class metrics check
        per_class = res["per_class_metrics"]
        for pii_cls in EVAL_PII_CLASSES:
            self.assertIn(pii_cls, per_class)
            m = per_class[pii_cls]
            for key in ["precision", "recall", "f1", "f1_score", "tp", "fp", "fn", "tn"]:
                self.assertIn(key, m, f"Missing {key} in per_class_metrics for {pii_cls}")

        # 4. Micro/Macro check
        self.assertIn("micro_average", res)
        self.assertIn("macro_average", res)
        for key in ["precision", "recall", "f1"]:
            self.assertIn(key, res["micro_average"])
            self.assertIn(key, res["macro_average"])


if __name__ == "__main__":
    unittest.main()
