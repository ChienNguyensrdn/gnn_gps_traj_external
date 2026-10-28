from __future__ import annotations

import csv
import argparse
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np

from hybrid.beliefmove.quantitative_support import (
    aggregate_poi_probabilities,
    bind_teacher_queries,
    build_quantitative_inputs,
    export,
    region_assignment,
)
from hybrid.beliefmove.experiment import prepare
from hybrid.beliefmove.regions import UNKNOWN_REGION
from hybrid.neural_cgm import ModelConfig, build_model, _torch


class BeliefMoveQuantitativeSupportTests(unittest.TestCase):
    def test_region_assignment_and_mass_aggregation(self) -> None:
        support = ["r0", "r1", "r2", UNKNOWN_REGION]
        assignment = region_assignment(
            ["p0", "p1", "p2", "p3"],
            {"p0": "r0", "p1": "r0", "p2": "r2"},
            support,
        )
        self.assertEqual(assignment.tolist(), [0, 0, 2, 3])
        probabilities = aggregate_poi_probabilities(
            np.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=np.float64), assignment, len(support)
        )
        np.testing.assert_allclose(probabilities, [[0.3, 0.0, 0.3, 0.4]])

    def test_inputs_preserve_query_ids_and_never_emit_target_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "val.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "trajectory_id", "POI_id", "UTC_time"])
                writer.writerows(
                    [
                        ["u", "tr", 0, "2024-01-01 08:00:00"],
                        ["u", "tr", 1, "2024-01-01 09:00:00"],
                        ["u", "tr", 2, "2024-01-01 10:00:00"],
                    ]
                )
            query_rows = [
                {
                    "query_id": "validation:q1",
                    "user_id": "u",
                    "trajectory_id": "tr",
                    "prefix_length": 1,
                    "query_time": "2024-01-01T09:00:00.000000Z",
                    "observed_prefix": [
                        {"observed_time": "2024-01-01T08:00:00.000000Z", "region": "r0"}
                    ],
                },
                {
                    "query_id": "validation:q2",
                    "user_id": "u",
                    "trajectory_id": "tr",
                    "prefix_length": 2,
                    "query_time": "2024-01-01T10:00:00.000000Z",
                    "observed_prefix": [
                        {"observed_time": "2024-01-01T08:00:00.000000Z", "region": "r0"},
                        {"observed_time": "2024-01-01T09:00:00.000000Z", "region": "r1"},
                    ],
                },
            ]
            inputs = build_quantitative_inputs(
                query_rows,
                csv_path,
                ["p0", "p1", "p2"],
                {"p0": "r0", "p1": "r1", "p2": "r2"},
            )
        self.assertEqual([row["query_id"] for row in inputs], ["validation:q1", "validation:q2"])
        self.assertEqual(inputs[0]["context_poi_ids"], ["p0"])
        self.assertEqual(inputs[0]["query_time"], "2024-01-01T09:00:00.000000Z")
        self.assertEqual(inputs[1]["context_poi_ids"], ["p0", "p1"])
        for row in inputs:
            self.assertNotIn("target_poi_id", row)
            self.assertNotIn("target_region", row)
            self.assertNotIn("label", row)

    def test_prepared_prefix_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            csv_path = Path(temporary) / "test.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["user_id", "trajectory_id", "POI_id", "UTC_time"])
                writer.writerows(
                    [["u", "tr", 0, "2024-01-01 08:00:00"], ["u", "tr", 1, "2024-01-01 09:00:00"]]
                )
            query = {
                "query_id": "test:q",
                "user_id": "u",
                "trajectory_id": "tr",
                "prefix_length": 1,
                "query_time": "2024-01-01T09:00:00.000000Z",
                "observed_prefix": [{"observed_time": "WRONG", "region": "r0"}],
            }
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                build_quantitative_inputs([query], csv_path, ["p0", "p1"], {"p0": "r0"})

    def test_tiny_checkpoint_export_is_all_prefix_and_target_free(self) -> None:
        torch = _torch()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            prepared = root / "prepared"
            (data / "getnext").mkdir(parents=True)
            (data / "neural_cgm").mkdir(parents=True)
            candidate_ids = ["p0", "p1"]
            (data / "candidate_ids.json").write_text(json.dumps(candidate_ids), encoding="utf-8")
            columns = [
                "user_id", "POI_id", "POI_catname", "latitude", "longitude", "trajectory_id", "UTC_time"
            ]
            for filename, trajectory in (("train.csv", "tr"), ("val.csv", "va"), ("test.csv", "te")):
                with (data / "getnext" / filename).open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(columns)
                    writer.writerows(
                        [
                            ["u", 0, "home", 10.0, 106.0, trajectory, "2024-01-01 08:00:00"],
                            ["u", 1, "work", 10.01, 106.01, trajectory, "2024-01-01 09:00:00"],
                            ["u", 0, "home", 10.0, 106.0, trajectory, "2024-01-01 10:00:00"],
                        ]
                    )
            config = ModelConfig(num_pois=2, num_users=1, poi_dim=2, user_dim=2, time_dim=2, hidden_dim=2)
            model = build_model(config)
            checkpoint_path = data / "neural_cgm" / "best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": asdict(config),
                    "user_map": {"u": 0},
                    "candidate_ids": candidate_ids,
                },
                checkpoint_path,
            )
            prepare(
                argparse.Namespace(
                    dataset="tist2015",
                    unit="Tokyo",
                    data_base=str(data),
                    output_root=str(prepared),
                    region_cell_meters=500.0,
                )
            )
            # Export must never open the held-out target artifact.
            (prepared / "targets" / "validation.jsonl").write_text("{malformed\n", encoding="utf-8")
            result = export(
                argparse.Namespace(
                    data_base=str(data),
                    prepared_root=str(prepared),
                    checkpoint=str(checkpoint_path),
                    output_root=None,
                    splits=["validation"],
                    batch_size=2,
                    top_k=2,
                    device="cpu",
                )
            )
            split_root = Path(result["splits"]["validation"]["output"])
            with np.load(split_root / "region_support.npz", allow_pickle=False) as values:
                self.assertEqual(values["query_ids"].tolist(), [
                    json.loads(line)["query_id"]
                    for line in (prepared / "queries" / "validation.jsonl").read_text().splitlines()
                ])
                self.assertEqual(values["probabilities"].shape[0], 2)
                np.testing.assert_allclose(values["probabilities"].sum(axis=1), 1.0, atol=1e-6)
            manifest = json.loads((split_root / "manifest.json").read_text())
            self.assertFalse(manifest["target_artifacts_read"])
            self.assertFalse(manifest["target_label_fields_emitted"])

            test_args = argparse.Namespace(
                data_base=str(data), prepared_root=str(prepared), checkpoint=str(checkpoint_path),
                output_root=None, results_root=str(root), protocol=str(root / "protocol.json"),
                splits=["test"], batch_size=2, top_k=2, device="cpu"
            )
            test_output = split_root.parent / "test"
            with self.assertRaisesRegex(FileNotFoundError, "requires protocol lock"):
                export(test_args)
            self.assertFalse(test_output.exists())

            protocol = {"protocol_name": "test", "version": "v2"}
            protocol_path = root / "protocol.json"
            protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
            protocol_hash = hashlib.sha256(
                json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            lock_path = root / "locks" / "protocol_lock.json"
            lock_path.parent.mkdir(parents=True)
            lock_payload = {
                "version": "beliefmove.protocol-lock.v2",
                "protocol_hash": protocol_hash,
                "pilot_report_hash": "b" * 64,
                "pilot_gate": "incomplete",
                "test_artifacts_read": False,
            }
            lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "pilot_gate is not ready"):
                export(test_args)
            self.assertFalse(test_output.exists())

            lock_payload["pilot_gate"] = "ready"
            lock_payload["protocol_hash"] = "a" * 64
            lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "protocol hash does not match"):
                export(test_args)
            self.assertFalse(test_output.exists())

            lock_payload["protocol_hash"] = protocol_hash
            lock_path.write_text(json.dumps(lock_payload), encoding="utf-8")
            # A corrupt target artifact demonstrates that the locked export
            # still never opens held-out labels.
            (prepared / "targets" / "test.jsonl").write_text("{malformed\n", encoding="utf-8")
            test_result = export(test_args)
            test_manifest = json.loads(
                (Path(test_result["splits"]["test"]["output"]) / "manifest.json").read_text()
            )
            self.assertEqual(test_manifest["protocol_lock"]["pilot_gate"], "ready")
            self.assertEqual(test_manifest["protocol_lock"]["protocol_hash"], protocol_hash)
            self.assertFalse(test_manifest["target_artifacts_read"])
            binding = bind_teacher_queries(
                argparse.Namespace(
                    data_base=str(data), prepared_root=str(prepared), checkpoint=str(checkpoint_path),
                    output_root=None, splits=["validation"]
                )
            )
            bound_path = Path(binding["splits"]["validation"]["output"])
            bound_rows = [json.loads(line) for line in bound_path.read_text().splitlines()]
            self.assertEqual(len(bound_rows), 2)
            self.assertTrue(all(row["candidate_regions"] for row in bound_rows))
            self.assertTrue(all("candidate_support_provenance" in row for row in bound_rows))
            self.assertTrue(all("target_poi_id" not in row for row in bound_rows))
            self.assertFalse(json.loads((bound_path.parent / "manifest.json").read_text())["target_artifacts_read"])
            # Exact rerun verifies immutable content instead of overwriting it.
            self.assertEqual(export(
                argparse.Namespace(
                    data_base=str(data), prepared_root=str(prepared), checkpoint=str(checkpoint_path),
                    output_root=None, splits=["validation"], batch_size=2, top_k=2, device="cpu"
                )
            )["gate"], "ready")


if __name__ == "__main__":
    unittest.main()
