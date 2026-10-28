from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from hybrid.beliefmove import (
    ACTIVITY_ONTOLOGY,
    DESTINATION_ONTOLOGY,
    INTENT_ONTOLOGY,
    PilotEvaluationError,
    StructuredBelief,
    evaluate_pilot,
    hash_text,
    ontology_hash,
)
from hybrid.beliefmove.pilot import ONTOLOGY_LABEL_VERSION
from hybrid.beliefmove.experiment import _pilot_evaluation


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    payload = value if isinstance(value, bytes) else canonical(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(labels, selected: str, probability: float = 0.9) -> dict[str, float]:
    remainder = (1.0 - probability) / (len(labels) - 1)
    return {label: probability if label == selected else remainder for label in labels}


class BeliefMovePilotTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "unit"
        self.root.mkdir(parents=True)
        self.dataset_hash = digest(b"dataset")
        self.mapping_hash = digest(b"mapping")
        self.support = ["r-a", "r-b"]
        self.queries = {
            "train": [self._query("train", "t1"), self._query("train", "t2")],
            "validation": [self._query("validation", "v1"), self._query("validation", "v2")],
        }
        self.targets = {
            "train": [self._target("train", "t1", "r-a"), self._target("train", "t2", "r-a")],
            "validation": [
                self._target("validation", "v1", "r-b"),
                self._target("validation", "v2", "r-b"),
            ],
        }
        split_hashes = {}
        for split in ("train", "validation"):
            query_path = self.root / "queries" / f"{split}.jsonl"
            target_path = self.root / "targets" / f"{split}.jsonl"
            self._write_jsonl(query_path, self.queries[split])
            self._write_jsonl(target_path, self.targets[split])
            split_hashes[split] = {
                "queries": file_hash(query_path),
                "targets": file_hash(target_path),
                "query_count": len(self.queries[split]),
            }
        # Malformed test artifacts prove the evaluator never opens them.
        self._write_text(self.root / "queries" / "test.jsonl", "{malformed")
        self._write_text(self.root / "targets" / "test.jsonl", "{malformed")
        self._write_json(
            self.root / "manifests" / "prepare_manifest.json",
            {
                "version": "fixture",
                "dataset": "tist2015",
                "unit": "Tokyo",
                "dataset_hash": self.dataset_hash,
                "ontology_hash": ontology_hash(),
                "candidate_regions": self.support,
                "split_hashes": split_hashes,
            },
        )
        self.labels_root = self.root / "ontology_labels"
        label_hashes = {}
        for split in ("train", "validation"):
            activity = "residential" if split == "train" else "work-education"
            destination = "home-residential" if split == "train" else "work-education"
            rows = [
                {
                    "query_id": query["query_id"],
                    "split": split,
                    "source_hash": query["source_hash"],
                    "mapping_hash": self.mapping_hash,
                    "activity": activity,
                    "destination": destination,
                }
                for query in self.queries[split]
            ]
            path = self.labels_root / f"{split}.jsonl"
            self._write_jsonl(path, rows)
            label_hashes[split] = file_hash(path)
        self._write_json(
            self.labels_root / "manifest.json",
            {
                "version": ONTOLOGY_LABEL_VERSION,
                "dataset": "tist2015",
                "unit": "Tokyo",
                "fit_split": "train",
                "ontology_hash": ontology_hash(),
                "mapping_hash": self.mapping_hash,
                "test_artifacts_read": False,
                "split_hashes": label_hashes,
            },
        )
        self.base = self.root / "teacher" / "model" / "pilot"
        self._cache(
            self.base,
            tag="pilot",
            selected={"train": ["t1", "t2"], "validation": ["v1", "v2"]},
            probability=0.9,
        )
        self.repeats = []
        for index, probability in enumerate((0.88, 0.90, 0.92), start=1):
            path = self.root / "teacher" / "model" / f"pilot-repeat-{index}"
            self._cache(
                path,
                tag=f"pilot-repeat-{index}",
                selected={"train": [], "validation": ["v1", "v2"]},
                probability=probability,
            )
            self.repeats.append(path)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _query(self, split: str, suffix: str) -> dict:
        identity = {"split": split, "id": suffix}
        return {
            "query_id": f"{split}:{suffix}",
            "dataset": "tist2015",
            "unit": "Tokyo",
            "split": split,
            "user_id": "u0",
            "trajectory_id": f"trajectory-{split}",
            "prefix_length": 1,
            "observed_prefix": [
                {"observed_time": "2024-01-01T00:00:00Z", "region": "r-a", "activity_hint": ""}
            ],
            "candidate_regions": list(self.support),
            "source_hash": digest(identity),
        }

    @staticmethod
    def _target(split: str, suffix: str, region: str) -> dict:
        return {
            "query_id": f"{split}:{suffix}",
            "split": split,
            "target_poi_id": f"poi-{suffix}",
            "target_region": region,
            "raw_target_region": region,
        }

    @staticmethod
    def _write_text(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding="utf-8")

    def _write_json(self, path: Path, value) -> None:
        self._write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    def _write_jsonl(self, path: Path, rows) -> None:
        self._write_text(path, "".join(canonical(row) + "\n" for row in rows))

    def _cache(self, path: Path, *, tag: str, selected: dict[str, list[str]], probability: float) -> None:
        query_lookup = {
            split: {row["query_id"].split(":", 1)[1]: row for row in self.queries[split]}
            for split in ("train", "validation")
        }
        selected_full = {
            split: [f"{split}:{suffix}" for suffix in suffixes]
            for split, suffixes in selected.items()
        }
        selection = {
            "version": "beliefmove.teacher-cache.v2",
            "dataset": "tist2015",
            "unit": "Tokyo",
            "cache_tag": tag,
            "seed": 42,
            "model": "fake:1b",
            "model_digest": "d" * 64,
            "prompt_contract_hash": digest(b"prompt-contract"),
            "dataset_hash": self.dataset_hash,
            "ontology_hash": ontology_hash(),
            "query_ids": selected_full,
            "query_hashes": {
                split: digest([query_lookup[split][suffix] for suffix in selected[split]])
                for split in ("train", "validation")
            },
            "requested_by_split": (
                {"train": 500, "validation": 200}
                if tag == "pilot"
                else {"train": 0, "validation": 50}
            ),
            "available_by_split": {
                split: len(self.queries[split]) for split in ("train", "validation")
            },
            "test_queries_read": False,
        }
        attempts = []
        beliefs = []
        for split in ("train", "validation"):
            for suffix in selected[split]:
                query = query_lookup[split][suffix]
                activity = "residential" if split == "train" else "work-education"
                destination = "home-residential" if split == "train" else "work-education"
                prompt = f"prompt:{query['query_id']}"
                model_identity = f"Fake:fake:1b@{'d' * 64}"
                raw_response = canonical({"fixture": query["query_id"]})
                belief = StructuredBelief.create(
                    query=canonical(query),
                    prompt=prompt,
                    model_identity=model_identity,
                    activity=distribution(ACTIVITY_ONTOLOGY, activity, probability),
                    intent=distribution(INTENT_ONTOLOGY, "new-zone", probability),
                    destination=distribution(DESTINATION_ONTOLOGY, destination, probability),
                    next_region=distribution(self.support, "r-b", probability),
                    confidence=0.8,
                )
                record = {
                    "query_id": query["query_id"],
                    "split": split,
                    "source_hash": digest(query),
                    "prompt_hash": belief.prompt_hash,
                    "response_format_hash": digest(
                        {
                            "schema": "beliefmove.teacher-response-schema.v2",
                            "candidate_regions": query["candidate_regions"],
                        }
                    ),
                    "model_hash": belief.model_hash,
                    "dataset_hash": self.dataset_hash,
                    "ontology_hash": ontology_hash(),
                    "physical_request": 1,
                    "prompt": prompt,
                    "raw_response": raw_response,
                    "raw_response_hash": hash_text(raw_response),
                    "raw_valid": True,
                    "usable_output": True,
                    "fallback": False,
                    "transport_error": "",
                    "normalization": {
                        "version": "beliefmove.teacher-l1-weights.v1",
                        "raw_sums": {
                            "activity": 1.0,
                            "intent": 1.0,
                            "destination": 1.0,
                            "next_region": 1.0,
                        },
                        "applied": False,
                        "canonical_probability_error": 0.0,
                    },
                    "belief": belief.to_dict(),
                }
                attempts.append(
                    {
                        key: record[key]
                        for key in (
                            "query_id",
                            "split",
                            "source_hash",
                            "prompt_hash",
                            "response_format_hash",
                            "model_hash",
                            "dataset_hash",
                            "ontology_hash",
                            "physical_request",
                        )
                    }
                )
                beliefs.append(record)
        manifest = {
            **selection,
            "selected_queries": len(beliefs),
            "completed_queries": len(beliefs),
            "physical_requests": len(beliefs),
            "gate": "ready",
        }
        self._write_json(path / "selection_manifest.json", selection)
        self._write_json(path / "manifest.json", manifest)
        self._write_jsonl(path / "attempts.jsonl", attempts)
        self._write_jsonl(path / "beliefs.jsonl", beliefs)

    def test_evaluation_is_raw_hash_bound_and_never_reads_test(self) -> None:
        result = evaluate_pilot(
            self.root,
            self.base,
            self.repeats,
            ontology_labels_root=self.labels_root,
        )
        self.assertEqual(result["gate"], "ready")
        self.assertFalse(result["test_artifacts_read"])
        self.assertGreater(result["teacher_activity_gain"], 0.0)
        self.assertGreater(result["teacher_destination_gain"], 0.0)
        self.assertLess(result["m2_validation_nll"], result["m1_validation_nll"])
        self.assertLessEqual(result["repeat_jsd_median"], 0.10)
        self.assertTrue(result["gates"]["m2_heldout_nll_improves"])
        self.assertEqual(result["m1_m2_validation"]["alpha_grid"], [0.0, 0.25, 0.5, 0.75, 1.0])
        partition = result["m1_m2_validation"]["validation_partition"]
        self.assertTrue(partition["disjoint"])
        self.assertEqual(partition["alpha_selection_queries"], 1)
        self.assertEqual(partition["heldout_evaluation_queries"], 1)
        self.assertNotEqual(
            partition["alpha_selection_query_hash"], partition["heldout_evaluation_query_hash"]
        )
        self.assertEqual(
            result["teacher_cache_hash"], file_hash(self.base / "beliefs.jsonl")
        )
        written = json.loads(
            (self.root / "manifests" / "pilot_evaluation.json").read_text(encoding="utf-8")
        )
        self.assertEqual(written["evaluation_hash"], result["evaluation_hash"])

    def test_missing_ontology_labels_fails_closed(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.root / "missing-labels",
                write=False,
            )

    def test_prepared_target_ontology_labels_are_consumed_directly(self) -> None:
        for split in ("train", "validation"):
            activity = "residential" if split == "train" else "work-education"
            destination = "home-residential" if split == "train" else "work-education"
            rows = [dict(row) for row in self.targets[split]]
            for row in rows:
                row["target_activity"] = activity
                row["target_destination"] = destination
            path = self.root / "targets" / f"{split}.jsonl"
            self._write_jsonl(path, rows)
            manifest_path = self.root / "manifests" / "prepare_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["split_hashes"][split]["targets"] = file_hash(path)
            self._write_json(manifest_path, manifest)

        result = evaluate_pilot(
            self.root,
            self.base,
            self.repeats,
            ontology_labels_root=self.root / "must-not-be-read",
            write=True,
        )
        self.assertEqual(result["hashes"]["ontology_labels"]["source"], "prepared-targets")
        self.assertEqual(result["gate"], "ready")
        verified = _pilot_evaluation(
            self.root,
            self.base,
            file_hash(self.base / "beliefs.jsonl"),
        )
        self.assertIsNotNone(verified)
        self.assertEqual(verified["evaluation_hash"], result["evaluation_hash"])

        summary_path = self.root / "manifests" / "pilot_evaluation.json"
        stale = json.loads(summary_path.read_text())
        stale["m2_validation_nll"] = stale["m2_validation_nll"] + 1.0
        self._write_json(summary_path, stale)
        self.assertIsNone(
            _pilot_evaluation(
                self.root,
                self.base,
                file_hash(self.base / "beliefs.jsonl"),
            )
        )

    def test_stale_ontology_mapping_hash_fails_closed(self) -> None:
        path = self.labels_root / "validation.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["mapping_hash"] = digest(b"different-mapping")
        self._write_jsonl(path, rows)
        manifest = json.loads((self.labels_root / "manifest.json").read_text())
        manifest["split_hashes"]["validation"] = file_hash(path)
        self._write_json(self.labels_root / "manifest.json", manifest)
        with self.assertRaisesRegex(PilotEvaluationError, "provenance mismatch"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_repeat_query_set_mismatch_fails_closed(self) -> None:
        selection_path = self.repeats[0] / "selection_manifest.json"
        selection = json.loads(selection_path.read_text())
        selection["query_ids"]["validation"] = ["validation:v1"]
        selection["query_hashes"]["validation"] = digest([self.queries["validation"][0]])
        self._write_json(selection_path, selection)
        # Keep cache content inconsistent intentionally: the evaluator must
        # reject the stale selection before producing a derived gate.
        with self.assertRaisesRegex(PilotEvaluationError, "budget contract|incomplete|unexpected"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_repeat_invalid_output_fails_closed_before_jsd(self) -> None:
        beliefs_path = self.repeats[0] / "beliefs.jsonl"
        rows = [json.loads(line) for line in beliefs_path.read_text().splitlines()]
        rows[0]["raw_valid"] = False
        self._write_jsonl(beliefs_path, rows)
        with self.assertRaisesRegex(
            PilotEvaluationError,
            "usable_output/raw_valid mismatch|invalid/fallback/transport-error",
        ):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_legacy_teacher_cache_version_fails_closed(self) -> None:
        for name in ("selection_manifest.json", "manifest.json"):
            path = self.base / name
            payload = json.loads(path.read_text())
            payload["version"] = "beliefmove.teacher-cache.v1"
            self._write_json(path, payload)
        with self.assertRaisesRegex(PilotEvaluationError, "teacher-cache.v2"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_missing_response_format_provenance_fails_closed(self) -> None:
        for name in ("attempts.jsonl", "beliefs.jsonl"):
            path = self.base / name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0].pop("response_format_hash")
            self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "response_format_hash"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_missing_normalization_provenance_fails_closed(self) -> None:
        path = self.base / "beliefs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0].pop("normalization")
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "missing normalization provenance"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_legacy_normalization_version_fails_closed(self) -> None:
        path = self.base / "beliefs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["normalization"]["version"] = "beliefmove.teacher-l1-weights.v0"
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "normalization version mismatch"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_usable_output_must_match_raw_valid(self) -> None:
        path = self.base / "beliefs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["usable_output"] = False
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "usable_output/raw_valid mismatch"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_canonical_probability_error_must_be_bounded(self) -> None:
        path = self.base / "beliefs.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["normalization"]["canonical_probability_error"] = 1e-4
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "canonical probability error"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_repeat_response_format_hash_must_match_base(self) -> None:
        replacement = "a" * 64
        for name in ("attempts.jsonl", "beliefs.jsonl"):
            path = self.repeats[0] / name
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["response_format_hash"] = replacement
            self._write_jsonl(path, rows)
        with self.assertRaisesRegex(PilotEvaluationError, "response_format_hash"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_budget_contract_mismatch_fails_closed(self) -> None:
        for name in ("selection_manifest.json", "manifest.json"):
            path = self.base / name
            payload = json.loads(path.read_text())
            payload["requested_by_split"]["train"] = 499
            self._write_json(path, payload)
        with self.assertRaisesRegex(PilotEvaluationError, "requested budget mismatch"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                write=False,
            )

    def test_alpha_endpoint_does_not_pass_strict_heldout_gate(self) -> None:
        target_path = self.root / "targets" / "validation.jsonl"
        targets = [dict(row, target_region="r-a", raw_target_region="r-a") for row in self.targets["validation"]]
        self._write_jsonl(target_path, targets)
        manifest_path = self.root / "manifests" / "prepare_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["split_hashes"]["validation"]["targets"] = file_hash(target_path)
        self._write_json(manifest_path, manifest)

        result = evaluate_pilot(
            self.root,
            self.base,
            self.repeats,
            ontology_labels_root=self.labels_root,
            write=False,
        )
        self.assertEqual(result["m1_m2_validation"]["selected_alpha"], 1.0)
        self.assertEqual(result["m2_validation_nll"], result["m1_validation_nll"])
        self.assertFalse(result["gates"]["m2_heldout_nll_improves"])
        self.assertEqual(result["gate"], "incomplete")

    def test_alpha_grid_is_not_tunable_through_the_api(self) -> None:
        with self.assertRaisesRegex(PilotEvaluationError, "alpha grid is frozen"):
            evaluate_pilot(
                self.root,
                self.base,
                self.repeats,
                ontology_labels_root=self.labels_root,
                alpha_grid=(0.0, 0.5, 1.0),
                write=False,
            )


if __name__ == "__main__":
    unittest.main()
