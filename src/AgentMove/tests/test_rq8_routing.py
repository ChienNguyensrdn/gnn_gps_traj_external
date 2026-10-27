import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from hybrid.rq8_routing import (
    budget_capped_mask,
    choose_threshold,
    load_always,
    metrics,
    oracle_mask,
    routed,
)


class RQ8RoutingTest(unittest.TestCase):
    def test_router_uses_llm_only_when_called(self):
        np.testing.assert_array_equal(routed([1, 4, 3], [2, 1, 1], [False, True, False]), [1, 1, 3])

    def test_threshold_respects_validation_budget(self):
        uncertainty = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = choose_threshold("entropy", uncertainty, np.asarray([1, 2, 4, 5]),
                                     np.asarray([1, 1, 1, 1]), [], 0.25, 4)
        self.assertLessEqual(float(np.mean(uncertainty > threshold)), 0.25)

    def test_test_distribution_shift_is_projected_to_hard_budget(self):
        uncertainty = np.asarray([0.9, 0.8, 0.7, 0.6])
        mask = budget_capped_mask("entropy", uncertainty, threshold=0.5, budget=0.25)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_margin_budget_cap_keeps_smallest_eligible_scores_stably(self):
        uncertainty = np.asarray([0.1, 0.1, 0.2, 0.3])
        mask = budget_capped_mask("margin", uncertainty, threshold=0.4, budget=0.5)
        np.testing.assert_array_equal(mask, [True, True, False, False])

    def test_cost_metrics_include_zero_for_uncalled_queries(self):
        rows = [{"latency_seconds": 2, "input_tokens": 10, "output_tokens": 2,
                 "api_calls": 1, "valid": True}] * 2
        result = metrics(np.asarray([1, 2]), np.asarray([True, False]), rows)
        self.assertEqual(result["llm_call_rate"], 0.5)
        self.assertEqual(result["selected_query_rate"], 0.5)
        self.assertEqual(result["api_requests_per_query"], 0.5)
        self.assertEqual(result["latency_mean"], 1.0)
        self.assertEqual(result["tokens_per_query"], 6.0)

    def test_physical_requests_are_not_confused_with_selected_rate(self):
        rows = [{"latency_seconds": 1, "input_tokens": 8, "output_tokens": 2,
                 "api_calls": 4, "valid": False},
                {"latency_seconds": 1, "input_tokens": 8, "output_tokens": 2,
                 "api_calls": 1, "valid": True}]
        result = metrics(np.asarray([1, 2]), np.asarray([True, False]), rows)
        self.assertEqual(result["selected_query_rate"], 0.5)
        self.assertEqual(result["api_requests_per_query"], 2.0)
        self.assertEqual(result["invalid_outputs_per_query"], 0.5)

    def test_oracle_calls_only_positive_gain_within_budget(self):
        mask = oracle_mask(np.asarray([1, 5, 2, 3]), np.asarray([2, 1, 2, 1]), 0.5)
        np.testing.assert_array_equal(mask, [False, True, False, True])

    def test_always_cache_rejects_duplicate_query_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "always.jsonl"
            path.write_text('{"query_id":"q1"}\n{"query_id":"q1"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate query_id=q1"):
                load_always(path, ["q1"])

    def test_always_cache_rejects_unexpected_query_ids(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "always.jsonl"
            path.write_text('{"query_id":"q1"}\n{"query_id":"q2"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected query_id=q2"):
                load_always(path, ["q1"])


if __name__ == "__main__": unittest.main()
