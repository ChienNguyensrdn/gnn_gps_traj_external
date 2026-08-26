import unittest

import numpy as np

from hybrid.rq8_routing import choose_threshold, metrics, routed


class RQ8RoutingTest(unittest.TestCase):
    def test_router_uses_llm_only_when_called(self):
        np.testing.assert_array_equal(routed([1, 4, 3], [2, 1, 1], [False, True, False]), [1, 1, 3])

    def test_threshold_respects_validation_budget(self):
        uncertainty = np.asarray([0.1, 0.2, 0.8, 0.9])
        threshold = choose_threshold("entropy", uncertainty, np.asarray([1, 2, 4, 5]),
                                     np.asarray([1, 1, 1, 1]), [], 0.25, 4)
        self.assertLessEqual(float(np.mean(uncertainty > threshold)), 0.25)

    def test_cost_metrics_include_zero_for_uncalled_queries(self):
        rows = [{"latency_seconds": 2, "input_tokens": 10, "output_tokens": 2}] * 2
        result = metrics(np.asarray([1, 2]), np.asarray([True, False]), rows)
        self.assertEqual(result["llm_call_rate"], 0.5)
        self.assertEqual(result["latency_mean"], 1.0)
        self.assertEqual(result["tokens_per_query"], 6.0)


if __name__ == "__main__": unittest.main()
