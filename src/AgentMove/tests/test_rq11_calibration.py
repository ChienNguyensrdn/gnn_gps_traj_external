import unittest

import numpy as np

from hybrid.rq11_calibration import (arrays_from_probabilities, fit_calibrators,
                                     normalize_log_scores, reliability, summarize,
                                     test_outputs as collect_test_outputs)


class RQ11CalibrationTests(unittest.TestCase):
    def test_temperature_preserves_ranking_and_normalization(self):
        scores = np.array([[3.0, 1.0, -1.0], [0.2, 0.1, 0.0]])
        cold = normalize_log_scores(scores, 0.5); warm = normalize_log_scores(scores, 2.0)
        np.testing.assert_allclose(cold.sum(axis=1), 1.0)
        np.testing.assert_array_equal(cold.argmax(axis=1), warm.argmax(axis=1))
        self.assertGreater(cold[0, 0], warm[0, 0])

    def test_temperature_is_selected_from_validation_grid(self):
        batches = [(np.array([[8.0, 0.0], [8.0, 0.0], [0.0, 8.0]]), np.array([0, 1, 1]))]
        selected, trace = fit_calibrators(iter(batches), [0.5, 1.0, 2.0, 4.0], 3)
        self.assertEqual(selected["identity"], 1.0)
        self.assertEqual(selected["nll"], 4.0)
        self.assertEqual(set(trace), {"nll", "brier", "ece"})

    def test_metrics_and_reliability_are_finite(self):
        probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.7, 0.3]])
        arrays = arrays_from_probabilities(probabilities, np.array([0, 1, 1]))
        metrics = summarize(arrays, 3)
        self.assertEqual(metrics["queries"], 3)
        self.assertTrue(np.isfinite(metrics["nll"]))
        self.assertEqual(sum(row["count"] for row in reliability(arrays["confidence"], arrays["top1_correct"], 3, True)), 3)

    def test_streamed_test_outputs_keep_only_scalar_arrays(self):
        batches = iter([(np.array([[2.0, 1.0], [0.0, 1.0]]), np.array([0, 1]))])
        outputs = collect_test_outputs(batches, {"identity": 1.0, "nll": 2.0})
        self.assertEqual(set(outputs), {"identity", "nll"})
        self.assertEqual(outputs["identity"]["labels"].shape, (2,))
        self.assertNotIn("probabilities", outputs["identity"])


if __name__ == "__main__": unittest.main()
