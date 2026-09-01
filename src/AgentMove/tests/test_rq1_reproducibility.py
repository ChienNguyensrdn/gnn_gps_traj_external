import json
import tempfile
import unittest
from pathlib import Path

from hybrid.rq1_reproducibility import load_quantitative, optional_baseline


class RQ1ReproducibilityTests(unittest.TestCase):
    def test_quantitative_requires_every_declared_seed(self):
        metrics = {name: 0.1 for name in ("recall@1", "recall@5", "recall@10", "mrr", "nll", "brier", "ece")}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in ("teachers/gru", "teachers/transformer", "students/none"):
                for seed in (42, 43):
                    path = root / relative / f"seed-{seed}" / "test.metrics.json"
                    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps({"metrics": metrics}))
            result = load_quantitative(root, [42, 43])
            self.assertEqual(result["teacher-gru"]["recall@1"]["mean"], 0.1)
            with self.assertRaises(FileNotFoundError): load_quantitative(root, [42, 43, 44])

    def test_bounded_gate_checks_completeness_and_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps({"is_complete_12_city": True, "query_limit": 200,
                                        "macro_average": {"acc@1": 0.1}, "protocol": "bounded"}))
            self.assertEqual(optional_baseline(path, 200)["status"], "ready-bounded")
            self.assertEqual(optional_baseline(path, 100)["status"], "incomplete-or-incompatible")


if __name__ == "__main__": unittest.main()
