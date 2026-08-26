import unittest

from hybrid.free_text_rerank import _metrics
from hybrid.rq9_aggregate import ranking_sensitivity
from hybrid.rq9_semantic_rerank import donor_histories, perturb, random_poi_context, shuffled


class RQ9SemanticTest(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"query_id": "q1", "user_id": "u1", "history": [[1, "A", "a"], [2, "B", "b"]], "context": [[3, "C", "c"]]},
            {"query_id": "q2", "user_id": "u2", "history": [[4, "D", "d"]], "context": [[5, "E", "e"]]},
        ]

    def test_shuffle_is_deterministic_and_preserves_items(self):
        left = shuffled(self.rows[0]["history"], "q1", 42, "memory")
        right = shuffled(self.rows[0]["history"], "q1", 42, "memory")
        self.assertEqual(left, right); self.assertCountEqual(left, self.rows[0]["history"])

    def test_random_user_uses_another_history(self):
        donors = donor_histories(self.rows); history, context = perturb(self.rows[0], "memory-random-user", donors, {}, 42)
        self.assertEqual(history, self.rows[1]["history"]); self.assertEqual(context, self.rows[0]["context"])

    def test_random_poi_preserves_timestamp_and_changes_identity(self):
        result = random_poi_context([[3, "C", "c"]], {"x": {"category": "X"}}, "q1", 42)
        self.assertEqual(result, [[3, "X", "x"]])

    def test_one_axis_none_preserves_other_axis(self):
        donors = donor_histories(self.rows)
        history, context = perturb(self.rows[0], "context-none", donors, {}, 42)
        self.assertEqual(history, self.rows[0]["history"]); self.assertEqual(context, [])

    def test_rq9_metric_schema_includes_candidate_recall(self):
        row = {"true_rank": 3, "true_in_top_k": True, "valid": True, "input_tokens": 10,
               "output_tokens": 2, "api_calls": 1, "latency_seconds": 0.5}
        self.assertEqual(_metrics([row])["candidate_recall"], 1.0)

    def test_ranking_sensitivity_detects_top1_change(self):
        left = [{"ranking_top_k": ["a", "b", "c"]}]
        right = [{"ranking_top_k": ["b", "a", "c"]}]
        result = ranking_sensitivity(left, right)
        self.assertEqual(result["top1_change_rate"], 1.0)
        self.assertEqual(result["ranking_change_rate"], 1.0)


if __name__ == "__main__": unittest.main()
