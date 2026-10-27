import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from hybrid.rq8_routing import (
    POLICIES,
    budget_capped_mask,
    choose_threshold,
    evaluate,
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

    def test_batch_seeds_compute_stage_arrays_once_per_split(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "routing.json"
            config.write_text(json.dumps({
                "target_call_rate": 0.25,
                "auto_quantile_candidates": 4,
                "entropy_thresholds": [],
                "margin_thresholds": [],
                "call_budgets": [0.25],
                "test_budget_enforcement": "deterministic_uncertainty_rank_hard_cap",
            }))
            calibration = root / "temperature.json"
            calibration.write_text(json.dumps({"temperature": {"temperature": 1.0}}))
            validation = root / "validation.jsonl"; validation.write_text("{}\n")
            test = root / "test.jsonl"; test.write_text("{}\n")
            validation_always = root / "validation.always.jsonl"
            test_always = root / "test.always.jsonl"
            rows = [{"query_id": f"q{index}", "user_id": f"u{index}"}
                    for index in range(4)]
            always_rows = [dict(row, true_rank=1, latency_seconds=1.0,
                                input_tokens=10, output_tokens=2, api_calls=1,
                                valid=True) for row in rows]
            payload = "".join(json.dumps(row) + "\n" for row in always_rows)
            validation_always.write_text(payload); test_always.write_text(payload)
            split = (rows, np.zeros((4, 2)), ["a", "b"])
            val_arrays = (np.asarray([1, 2, 3, 4]), np.asarray([0.1, 0.2, 0.8, 0.9]),
                          np.asarray([0.9, 0.8, 0.2, 0.1]))
            test_arrays = (np.asarray([1, 2, 3, 4]), np.asarray([0.1, 0.2, 0.8, 0.9]),
                           np.asarray([0.9, 0.8, 0.2, 0.1]))
            args = SimpleNamespace(
                config=str(config), calibration=str(calibration), validation=str(validation),
                test=str(test), validation_always=str(validation_always),
                test_always=str(test_always), limit=4, seed=42,
                seeds=[42, 43], output_root=str(root / "routing"), output_dir=None)
            with patch("hybrid.rq8_routing.load_split", side_effect=[split, split]), \
                    patch("hybrid.rq8_routing.stage_arrays",
                          side_effect=[val_arrays, test_arrays]) as mocked_stage:
                result = evaluate(args)
            self.assertEqual(mocked_stage.call_count, 2)
            self.assertEqual([row["seed"] for row in result], [42, 43])
            deterministic = []
            for seed in (42, 43):
                folder = root / f"routing/seed-{seed}"
                self.assertTrue((folder / "rq8.metrics.json").is_file())
                deterministic.append(np.load(folder / "entropy.test.predictions.npz")["called_llm"])
                random = np.load(folder / "random-budget-matched.test.predictions.npz")["called_llm"]
                expected = np.zeros(4, dtype=np.int8)
                expected[np.random.default_rng(seed).permutation(4)[:1]] = 1
                np.testing.assert_array_equal(random, expected)
            np.testing.assert_array_equal(deterministic[0], deterministic[1])
            single_args = SimpleNamespace(**{**vars(args), "seeds": None,
                                             "output_root": None,
                                             "output_dir": str(root / "single"),
                                             "seed": 42})
            with patch("hybrid.rq8_routing.load_split", side_effect=[split, split]), \
                    patch("hybrid.rq8_routing.stage_arrays",
                          side_effect=[val_arrays, test_arrays]) as single_stage:
                evaluate(single_args)
            self.assertEqual(single_stage.call_count, 2)
            for policy in POLICIES:
                batch_arrays = np.load(root / f"routing/seed-42/{policy}.test.predictions.npz")
                single_arrays = np.load(root / f"single/{policy}.test.predictions.npz")
                for key in ("query_id", "user_id", "ranks", "called_llm"):
                    np.testing.assert_array_equal(batch_arrays[key], single_arrays[key])


if __name__ == "__main__": unittest.main()
