import json
import tempfile
import unittest
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from hybrid.neural_cgm import ModelConfig, _torch, build_model
from hybrid.rq12_aggregate import load_routing
from hybrid.rq12_efficiency import benchmark


class RQ12EfficiencyTests(unittest.TestCase):
    def test_cpu_smoke_benchmark_writes_reproducibility_metadata(self):
        torch = _torch(); config = ModelConfig(num_pois=3, num_users=2, hidden_dim=8)
        model = build_model(config)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); checkpoint = root / "best.pt"; test_csv = root / "test.csv"
            metrics = root / "metrics.json"; output = root / "rq12.json"
            torch.save({"model_state": model.state_dict(), "config": asdict(config),
                        "user_map": {"u0": 0, "u1": 1}, "candidate_ids": ["a", "b", "c"]}, checkpoint)
            rows = []
            for trajectory, user in (("t0", "u0"), ("t1", "u1")):
                for index, poi in enumerate((0, 1, 2)):
                    rows.append({"trajectory_id": trajectory, "user_id": user, "POI_id": poi,
                                 "UTC_time": f"2024-01-01 {8 + index:02d}:00:00"})
            pd.DataFrame(rows).to_csv(test_csv, index=False)
            metrics.write_text(json.dumps({"metrics": {"recall@1": .1, "recall@5": .2,
                                                         "recall@10": .3, "mrr": .15}}))
            result = benchmark(Namespace(checkpoint=str(checkpoint), test_csv=str(test_csv),
                quality_metrics=str(metrics), quality_variant=None, output=str(output), variant="smoke",
                protocol="last-query", train_csv=None, rq7_metrics=None, batch_size=2, warmup_batches=1,
                repeats=2, max_batches=None, device="cpu", seed=42, smoothing=1.0))
            self.assertTrue(output.is_file()); self.assertEqual(result["timing"]["measured_queries"], 4)
            self.assertGreater(result["timing"]["throughput_queries_per_second"], 0)
            self.assertEqual(result["hardware"]["device"], "cpu")

    def test_routing_source_must_be_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rq8.json"
            path.write_text(json.dumps({"gate": "ready", "limit": 200, "policies": {}}))
            with self.assertRaises(ValueError): load_routing(path)


if __name__ == "__main__": unittest.main()
