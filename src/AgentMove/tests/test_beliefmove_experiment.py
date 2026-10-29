from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hybrid.beliefmove.experiment import (
    ACTIVITY_ONTOLOGY,
    DESTINATION_ONTOLOGY,
    INTENT_ONTOLOGY,
    _candidate_id,
    _prefix_queries,
    _read_split,
    build_pilot_report,
    build_smoke_report,
    build_teacher_prompt,
    build_teacher_response_format,
    lock,
    main,
    parse_teacher_response,
    prepare,
    recover_dangling,
    status,
    teacher_cache,
)
from hybrid.beliefmove.regions import UNKNOWN_REGION
from hybrid.beliefmove.ontology import ontology_hash
from hybrid.beliefmove.schema import StructuredBelief


CSV_COLUMNS = (
    "user_id",
    "POI_id",
    "POI_catname",
    "latitude",
    "longitude",
    "trajectory_id",
    "UTC_time",
)


class FakeLLM:
    def __init__(self, *, malformed: bool = False) -> None:
        self.malformed = malformed
        self.prompts: list[str] = []
        self.response_formats: list[dict | None] = []
        self.calls = 0
        self.last_call_stats = {}

    def reset_call_stats(self) -> None:
        self.last_call_stats = {}

    def get_response_once(self, prompt: str, *, response_format=None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        self.response_formats.append(response_format)
        self.last_call_stats = {
            "latency_seconds": 0.01,
            "input_tokens": 25,
            "output_tokens": 10,
            "token_source": "fake",
        }
        if self.malformed:
            return "not-json"
        schema_text = prompt.split("Fixed output schema: ", 1)[1].split("\nObserved causal input:", 1)[0]
        schema = json.loads(schema_text)

        def uniform(keys):
            return {key: 1.0 / len(keys) for key in keys}

        return json.dumps(
            {
                "activity": uniform(list(schema["activity"])),
                "intent": uniform(list(schema["intent"])),
                "destination": uniform(list(schema["destination"])),
                "next_region": uniform(list(schema["next_region"])),
                "confidence": 0.75,
            }
        )


class BeliefMoveExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data"
        self.output = self.root / "results" / "tist2015" / "Tokyo"
        (self.data / "getnext").mkdir(parents=True)
        (self.data / "neural_cgm").mkdir(parents=True)
        (self.data / "candidate_ids.json").write_text(
            json.dumps(["poi-0", "poi-1", "poi-2"]), encoding="utf-8"
        )
        (self.data / "neural_cgm" / "best.pt").write_bytes(b"checkpoint")
        self._write_csv(
            "train.csv",
            [
                ("u0", 0, "home", 10.000, 106.000, "tr0", "2024-01-01T08:00:00Z"),
                ("u0", 1, "work", 10.010, 106.010, "tr0", "2024-01-01T09:00:00Z"),
                ("u0", 0, "home", 10.000, 106.000, "tr0", "2024-01-01T10:00:00Z"),
            ],
        )
        self._write_csv(
            "val.csv",
            [
                ("u0", 0, "home", 10.000, 106.000, "va0", "2024-01-02T08:00:00Z"),
                ("u0", 1, "work", 10.010, 106.010, "va0", "2024-01-02T09:00:00Z"),
                ("u0", 0, "home", 10.000, 106.000, "va0", "2024-01-02T10:00:00Z"),
            ],
        )
        self._write_csv(
            "test.csv",
            [
                ("u0", 0, "home", 10.000, 106.000, "te0", "2024-01-03T08:00:00Z"),
                # poi-2 and this coordinate occur only in test.
                ("u0", 2, "unknown", 20.000, 110.000, "te0", "2024-01-03T09:00:00Z"),
                ("u0", 0, "home", 10.000, 106.000, "te0", "2024-01-03T10:00:00Z"),
            ],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(self, name: str, rows) -> None:
        self._write_csv_at(self.data / "getnext" / name, rows)

    def _write_csv_at(self, path: Path, rows) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

    def _unit_args(self, command: str = "prepare") -> argparse.Namespace:
        return argparse.Namespace(
            command=command,
            dataset="tist2015",
            unit="Tokyo",
            data_base=str(self.data),
            output_root=str(self.output),
            region_cell_meters=500.0,
        )

    def _teacher_args(self, tag: str = "pilot") -> argparse.Namespace:
        return argparse.Namespace(
            command="teacher-cache",
            dataset="tist2015",
            unit="Tokyo",
            data_base=str(self.data),
            output_root=str(self.output),
            model="fake:1b",
            platform="Fake",
            model_digest="d" * 64,
            cache_tag=tag,
            train_limit=1,
            validation_limit=1,
            seed=42,
        )

    def _pilot_teacher_args(self, *, seed: int = 42) -> argparse.Namespace:
        args = self._teacher_args("pilot")
        args.train_limit = 500
        args.validation_limit = 200
        args.seed = seed
        return args

    def _recovery_args(self) -> argparse.Namespace:
        return argparse.Namespace(
            command="recover-dangling",
            dataset="tist2015",
            unit="Tokyo",
            data_base=str(self.data),
            output_root=str(self.output),
            model="fake:1b",
            platform="Fake",
            model_digest="d" * 64,
            query_root=None,
        )

    def _interrupted_base_pilot(self) -> tuple[Path, FakeLLM, bytes, bytes]:
        prepare(self._unit_args())
        fake = FakeLLM()
        result = teacher_cache(
            self._pilot_teacher_args(), llm_factory=lambda *_: fake
        )
        cache = Path(result["cache"])
        beliefs_path = cache / "beliefs.jsonl"
        belief_lines = beliefs_path.read_bytes().splitlines(keepends=True)
        self.assertGreaterEqual(len(belief_lines), 2)
        beliefs_path.write_bytes(b"".join(belief_lines[:-1]))
        (cache / "manifest.json").unlink()
        return (
            cache,
            fake,
            (cache / "attempts.jsonl").read_bytes(),
            beliefs_path.read_bytes(),
        )

    def test_prepare_separates_targets_and_fits_regions_from_train_only(self) -> None:
        result = prepare(self._unit_args())
        self.assertEqual(result["gate"], "ready")
        query_rows = [
            json.loads(line)
            for line in (self.output / "queries" / "test.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        target_rows = [
            json.loads(line)
            for line in (self.output / "targets" / "test.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(len(query_rows), 2)
        self.assertEqual(len(query_rows), len(target_rows))
        self.assertNotIn("target_poi_id", query_rows[0])
        self.assertNotIn("target_region", query_rows[0])
        self.assertNotIn("true_id", query_rows[0])
        self.assertNotIn("target_time", query_rows[0])
        self.assertIn("query_time", query_rows[0])
        self.assertIn("target_poi_id", target_rows[0])
        self.assertIn("target_activity", target_rows[0])
        self.assertIn("target_destination", target_rows[0])
        self.assertIn("target_intent", target_rows[0])
        poi_regions = json.loads(
            (self.output / "representation" / "poi_region_map.json").read_text(encoding="utf-8")
        )["mapping"]
        self.assertNotIn("poi-2", poi_regions)
        self.assertEqual(query_rows[1]["observed_prefix"][-1]["region"], UNKNOWN_REGION)
        manifest = json.loads(
            (self.output / "manifests" / "prepare_manifest.json").read_text(encoding="utf-8")
        )
        self.assertTrue(manifest["leakage_contract"]["target_artifacts_separate"])
        self.assertIn("test", manifest["split_hashes"])
        self.assertEqual(manifest["ontology_hash"], ontology_hash())
        # Exact rerun is a verification, not a rewrite.
        self.assertEqual(prepare(self._unit_args())["queries"], result["queries"])

    def test_chronology_uses_normalized_utc_not_lexicographic_text(self) -> None:
        path = self.root / "mixed-offsets.csv"
        self._write_csv_at(
            path,
            [
                # Lexicographically this 2023 row precedes the next row, but
                # 01:00+02:00 is 23:00Z and therefore the actual first event.
                ("u0", 0, "home", 10.000, 106.000, "tr0", "2023-12-31T23:30:00Z"),
                ("u0", 1, "work", 10.010, 106.010, "tr0", "2024-01-01T01:00:00+02:00"),
                ("u0", 2, "food", 10.020, 106.020, "tr0", "2024-01-01 00:00:00"),
            ],
        )
        rows = _read_split(path, ("poi-0", "poi-1", "poi-2"))
        queries, _ = _prefix_queries(
            "tist2015",
            "Tokyo",
            "train",
            rows,
            {"poi-0": "r0", "poi-1": "r1", "poi-2": "r2"},
            ("r0", "r1", "r2"),
        )
        self.assertEqual(queries[0]["observed_prefix"][0]["region"], "r1")
        self.assertEqual(queries[0]["observed_prefix"][0]["observed_time"], "2023-12-31T23:00:00.000000Z")
        self.assertEqual(queries[0]["query_time"], "2023-12-31T23:30:00.000000Z")

    def test_candidate_id_lookup_preserves_literal_before_numeric_index(self) -> None:
        candidate_ids = ("1", "venue-1", "venue-2")
        candidate_lookup = frozenset(candidate_ids)
        self.assertEqual(_candidate_id("1", candidate_ids, candidate_lookup), "1")
        self.assertEqual(_candidate_id("2.0", candidate_ids, candidate_lookup), "venue-2")
        self.assertEqual(_candidate_id("99", candidate_ids, candidate_lookup), "99")
        self.assertEqual(_candidate_id("unknown", candidate_ids, candidate_lookup), "unknown")

    def test_timestamp_parser_fails_closed_and_timezone_is_last_resort(self) -> None:
        invalid = self.root / "invalid-time.csv"
        self._write_csv_at(
            invalid,
            [("u0", 0, "home", 10.0, 106.0, "tr0", "not-a-timestamp")],
        )
        with self.assertRaisesRegex(ValueError, "unparseable timestamp"):
            _read_split(invalid, ("poi-0",))

        timezone_only = self.root / "timezone-only.csv"
        columns = tuple(value for value in CSV_COLUMNS if value != "UTC_time") + ("timezone",)
        with timezone_only.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerow(("u0", 0, "home", 10.0, 106.0, "tr0", "2024-01-01 08:00:00"))
        rows = _read_split(timezone_only, ("poi-0",))
        self.assertEqual(rows[0]["observed_time"], "2024-01-01T08:00:00.000000Z")

    def test_teacher_cache_is_train_validation_only_strict_and_resumable(self) -> None:
        prepare(self._unit_args())
        # A corrupt test target proves that the collector never opens it.
        (self.output / "targets" / "test.jsonl").write_text("not-json\n", encoding="utf-8")
        fake = FakeLLM()
        factory = lambda model, platform: fake
        result = teacher_cache(self._teacher_args(), llm_factory=factory)
        self.assertEqual(result["gate"], "ready")
        self.assertEqual(fake.calls, 2)
        self.assertEqual(len(fake.prompts), 2)
        for prompt in fake.prompts:
            self.assertIn("Observed causal input", prompt)
            self.assertIn("query_time", prompt)
            self.assertNotIn("target_time", prompt)
            self.assertNotIn("target_poi_id", prompt)
            self.assertNotIn("target_region", prompt)
        self.assertTrue(all(item and item["type"] == "json_schema" for item in fake.response_formats))
        cache = Path(result["cache"])
        beliefs = [json.loads(line) for line in (cache / "beliefs.jsonl").read_text().splitlines()]
        self.assertEqual({row["split"] for row in beliefs}, {"train", "validation"})
        self.assertTrue(all(row["raw_valid"] and not row["fallback"] for row in beliefs))
        for row in beliefs:
            StructuredBelief.from_dict(row["belief"])
        # Immutable resume makes no additional physical calls.
        resumed = teacher_cache(self._teacher_args(), llm_factory=factory)
        self.assertEqual(resumed["new_requests"], 0)
        self.assertEqual(fake.calls, 2)

    def test_invalid_teacher_json_is_recorded_as_fallback_without_retry(self) -> None:
        prepare(self._unit_args())
        fake = FakeLLM(malformed=True)
        result = teacher_cache(self._teacher_args("malformed"), llm_factory=lambda *_: fake)
        self.assertEqual(fake.calls, 2)
        cache = Path(result["cache"])
        beliefs = [json.loads(line) for line in (cache / "beliefs.jsonl").read_text().splitlines()]
        self.assertTrue(all(not row["raw_valid"] and row["fallback"] for row in beliefs))
        self.assertTrue(all(row["physical_request"] == 1 for row in beliefs))
        manifest = json.loads((cache / "manifest.json").read_text())
        self.assertEqual(manifest["fallbacks"], 2)
        self.assertEqual(manifest["collection_gate"], "ready")
        self.assertEqual(manifest["interface_gate"], "incomplete")
        self.assertEqual(manifest["gate"], "incomplete")

    def test_teacher_cache_resume_revalidates_attempt_response_format_hash(self) -> None:
        prepare(self._unit_args())
        fake = FakeLLM()
        result = teacher_cache(self._teacher_args(), llm_factory=lambda *_: fake)
        attempts_path = Path(result["cache"]) / "attempts.jsonl"
        attempts = [
            json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines()
        ]
        attempts[0]["response_format_hash"] = "0" * 64
        attempts_path.write_text(
            "".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            ValueError, "immutable teacher attempt conflict.*response_format_hash"
        ):
            teacher_cache(self._teacher_args(), llm_factory=lambda *_: fake)
        self.assertEqual(fake.calls, 2)

    def test_teacher_cache_require_gate_fails_after_writing_invalid_diagnostics(self) -> None:
        prepare(self._unit_args())
        args = self._teacher_args("smoke")
        args.require_gate = True
        with self.assertRaisesRegex(SystemExit, "3"):
            teacher_cache(args, llm_factory=lambda *_: FakeLLM(malformed=True))
        manifests = list((self.output / "teacher").rglob("manifest.json"))
        self.assertEqual(len(manifests), 1)
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifest["collection_gate"], "ready")
        self.assertEqual(manifest["interface_gate"], "incomplete")
        self.assertEqual(manifest["raw_valid"], 0)

    def test_recover_dangling_appends_deterministic_pilot_fallback_without_llm_call(self) -> None:
        cache, fake, attempts_before, beliefs_before = self._interrupted_base_pilot()
        selection_before = (cache / "selection_manifest.json").read_bytes()
        calls_before = fake.calls
        with mock.patch(
            "hybrid.beliefmove.experiment._make_llm",
            side_effect=AssertionError("recovery must never construct an LLM client"),
        ):
            result = recover_dangling(self._recovery_args())
        self.assertEqual(result["gate"], "ready")
        self.assertEqual(result["new_recoveries"], 1)
        self.assertEqual(result["llm_calls"], 0)
        self.assertEqual(result["request_budget_already_consumed"], 1)
        self.assertEqual(fake.calls, calls_before)
        self.assertEqual((cache / "attempts.jsonl").read_bytes(), attempts_before)
        self.assertEqual((cache / "selection_manifest.json").read_bytes(), selection_before)
        self.assertTrue((cache / "beliefs.jsonl").read_bytes().startswith(beliefs_before))
        self.assertFalse((cache / "manifest.json").exists())

        attempts = {
            row["query_id"]: row
            for row in map(json.loads, (cache / "attempts.jsonl").read_text().splitlines())
        }
        beliefs = {
            row["query_id"]: row
            for row in map(json.loads, (cache / "beliefs.jsonl").read_text().splitlines())
        }
        recovered_id = result["recovered_query_ids"][0]
        recovered = beliefs[recovered_id]
        self.assertEqual(recovered["transport_error"], "")
        self.assertEqual(
            recovered["fallback_reason"],
            "IndeterminateOutcome: collector stopped after durable attempt and before durable belief",
        )
        self.assertFalse(recovered["recovery"]["response_available"])
        self.assertFalse(recovered["recovery"]["usage_known"])
        self.assertEqual(recovered["recovery"]["request_outcome"], "indeterminate")
        self.assertTrue(recovered["recovery"]["request_budget_consumed"])
        self.assertEqual(recovered["physical_request"], 1)
        self.assertIsNone(recovered["latency_seconds"])
        self.assertIsNone(recovered["input_tokens"])
        self.assertIsNone(recovered["output_tokens"])
        self.assertEqual(
            recovered["recovery"]["attempt_hash"],
            hashlib.sha256(
                json.dumps(
                    attempts[recovered_id],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        )
        belief = StructuredBelief.from_dict(recovered["belief"])
        self.assertEqual(belief.confidence, 0.0)

        # Recovery is idempotent, and normal collection resumes without a
        # second physical request for the recovered query.
        second = recover_dangling(self._recovery_args())
        self.assertEqual(second["new_recoveries"], 0)
        resumed = teacher_cache(
            self._pilot_teacher_args(),
            llm_factory=lambda *_: (_ for _ in ()).throw(
                AssertionError("resume must not issue another physical request")
            ),
        )
        self.assertEqual(resumed["new_requests"], 0)
        manifest = json.loads((cache / "manifest.json").read_text())
        self.assertEqual(manifest["fallbacks"], 1)
        self.assertEqual(manifest["transport_errors"], 0)

    def test_recover_dangling_fails_closed_on_attempt_provenance_tamper(self) -> None:
        cache, _, _, beliefs_before = self._interrupted_base_pilot()
        attempts_path = cache / "attempts.jsonl"
        attempts = list(map(json.loads, attempts_path.read_text().splitlines()))
        attempts[-1]["prompt_hash"] = "0" * 64
        attempts_path.write_text(
            "".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "attempt provenance mismatch.*prompt_hash"):
            recover_dangling(self._recovery_args())
        self.assertEqual((cache / "beliefs.jsonl").read_bytes(), beliefs_before)

    def test_recover_dangling_rejects_tampered_recovery_distribution(self) -> None:
        cache, _, _, _ = self._interrupted_base_pilot()
        result = recover_dangling(self._recovery_args())
        recovered_id = result["recovered_query_ids"][0]
        beliefs_path = cache / "beliefs.jsonl"
        records = list(map(json.loads, beliefs_path.read_text().splitlines()))
        recovered = next(row for row in records if row["query_id"] == recovered_id)
        split = recovered["split"]
        query = next(
            row
            for row in map(
                json.loads,
                (self.output / "queries" / f"{split}.jsonl").read_text().splitlines(),
            )
            if row["query_id"] == recovered_id
        )
        original = StructuredBelief.from_dict(recovered["belief"])
        recovered["belief"] = StructuredBelief.create(
            query=json.dumps(
                query,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            prompt=recovered["prompt"],
            model_identity=f"Fake:fake:1b@{'d' * 64}",
            activity={key: 1.0 / len(ACTIVITY_ONTOLOGY) for key in ACTIVITY_ONTOLOGY},
            intent=original.intent,
            destination=original.destination,
            next_region=original.next_region,
            confidence=0.0,
        ).to_dict()
        beliefs_path.write_text(
            "".join(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
                for row in records
            ),
            encoding="utf-8",
        )
        tampered = beliefs_path.read_bytes()

        with self.assertRaisesRegex(ValueError, "exact deterministic fallback"):
            recover_dangling(self._recovery_args())
        self.assertEqual(beliefs_path.read_bytes(), tampered)

    def test_recover_dangling_rejects_multiple_dangling_attempts(self) -> None:
        cache, _, _, _ = self._interrupted_base_pilot()
        beliefs_path = cache / "beliefs.jsonl"
        lines = beliefs_path.read_bytes().splitlines(keepends=True)
        self.assertGreaterEqual(len(lines), 2)
        beliefs_path.write_bytes(b"".join(lines[:-1]))
        damaged = beliefs_path.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "multiple dangling attempts"):
            recover_dangling(self._recovery_args())
        self.assertEqual(beliefs_path.read_bytes(), damaged)

    def test_recover_dangling_rejects_active_collector_and_nonpilot_cache(self) -> None:
        cache, _, _, beliefs_before = self._interrupted_base_pilot()
        lock = (cache / ".collector.lock").open("a+", encoding="utf-8")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            with self.assertRaisesRegex(RuntimeError, "active collector"):
                recover_dangling(self._recovery_args())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        self.assertEqual((cache / "beliefs.jsonl").read_bytes(), beliefs_before)

        # A smoke cache cannot be selected for recovery even when interrupted.
        other_output = self.root / "smoke-only"
        self.output = other_output
        prepare(self._unit_args())
        smoke_args = self._teacher_args("smoke")
        smoke_args.train_limit = 1
        smoke_args.validation_limit = 1
        smoke_result = teacher_cache(smoke_args, llm_factory=lambda *_: FakeLLM())
        smoke_cache = Path(smoke_result["cache"])
        smoke_lines = (smoke_cache / "beliefs.jsonl").read_bytes().splitlines(keepends=True)
        (smoke_cache / "beliefs.jsonl").write_bytes(b"".join(smoke_lines[:-1]))
        (smoke_cache / "manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "exactly one matching base cache tagged 'pilot'"):
            recover_dangling(self._recovery_args())

    def test_recover_dangling_rejects_ambiguous_base_caches(self) -> None:
        prepare(self._unit_args())
        for seed in (42, 43):
            result = teacher_cache(
                self._pilot_teacher_args(seed=seed), llm_factory=lambda *_: FakeLLM()
            )
            cache = Path(result["cache"])
            lines = (cache / "beliefs.jsonl").read_bytes().splitlines(keepends=True)
            (cache / "beliefs.jsonl").write_bytes(b"".join(lines[:-1]))
            (cache / "manifest.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "observed=2"):
            recover_dangling(self._recovery_args())

    def test_smoke_report_requires_all_three_unambiguous_strict_caches(self) -> None:
        results_root = self.root / "smoke-results"
        digest = "a" * 64
        for dataset, unit in (
            ("tist2015", "Tokyo"),
            ("www2019", "Shanghai"),
            ("yjmob100k", "YJMob"),
        ):
            cache = results_root / dataset / unit / "teacher" / "fake" / "cache"
            cache.mkdir(parents=True)
            dataset_hash = hashlib.sha256(f"dataset:{dataset}".encode()).hexdigest()
            ontology_digest = hashlib.sha256(b"ontology").hexdigest()
            query_ids = {
                split: [f"{split}:{dataset}:{index}" for index in range(20)]
                for split in ("train", "validation")
            }
            selection = {
                "version": "beliefmove.teacher-cache.v2",
                "dataset": dataset,
                "unit": unit,
                "cache_tag": "smoke",
                "model": "fake:1b",
                "model_digest": digest,
                "dataset_hash": dataset_hash,
                "ontology_hash": ontology_digest,
                "requested_by_split": {"train": 20, "validation": 20},
                "available_by_split": {"train": 30, "validation": 25},
                "selected_by_split": {"train": 20, "validation": 20},
                "query_ids": query_ids,
                "test_queries_read": False,
            }
            manifest = {
                **selection,
                # Deliberately bogus derived counters: the smoke report must
                # recompute its gate from the journals below.
                "selected_queries": 0,
                "completed_queries": 0,
                "raw_valid": 0,
                "fallbacks": 40,
                "transport_errors": 40,
                "gate": "incomplete",
            }
            (cache / "selection_manifest.json").write_text(json.dumps(selection), encoding="utf-8")
            (cache / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            attempts, beliefs = [], []
            model_identity = f"Fake:fake:1b@{digest}"
            for split, ids in query_ids.items():
                for query_id in ids:
                    source = f"source:{query_id}"
                    prompt = f"prompt:{query_id}"
                    raw_response = "{}"
                    provenance = {
                        "query_id": query_id,
                        "split": split,
                        "source_hash": hashlib.sha256(source.encode()).hexdigest(),
                        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
                        "response_format_hash": hashlib.sha256(
                            f"format:{query_id}".encode()
                        ).hexdigest(),
                        "model_hash": hashlib.sha256(model_identity.encode()).hexdigest(),
                        "dataset_hash": dataset_hash,
                        "ontology_hash": ontology_digest,
                        "physical_request": 1,
                    }
                    belief = StructuredBelief.create(
                        query=source,
                        prompt=prompt,
                        model_identity=model_identity,
                        activity={key: 1.0 for key in ACTIVITY_ONTOLOGY},
                        intent={key: 1.0 for key in INTENT_ONTOLOGY},
                        destination={key: 1.0 for key in DESTINATION_ONTOLOGY},
                        next_region={"region": 1.0},
                        confidence=0.5,
                    ).to_dict()
                    attempts.append(provenance)
                    beliefs.append(
                        {
                            **provenance,
                            "prompt": prompt,
                            "raw_response": raw_response,
                            "raw_response_hash": hashlib.sha256(
                                raw_response.encode()
                            ).hexdigest(),
                            "raw_valid": True,
                            "usable_output": True,
                            "fallback": False,
                            "transport_error": "",
                            "normalization": {
                                "version": "beliefmove.teacher-l1-weights.v1",
                                "canonical_probability_error": 0.0,
                            },
                            "belief": belief,
                        }
                    )
            (cache / "attempts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in attempts), encoding="utf-8"
            )
            (cache / "beliefs.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in beliefs), encoding="utf-8"
            )

        report = build_smoke_report(
            results_root,
            model="fake:1b",
            model_digest=digest,
        )
        self.assertEqual(report["gate"], "ready")
        self.assertTrue(report["all_units_present"])
        self.assertFalse(report["test_artifacts_read"])

        tokyo_beliefs = next(
            (results_root / "tist2015" / "Tokyo").rglob("beliefs.jsonl")
        )
        rows = [json.loads(line) for line in tokyo_beliefs.read_text(encoding="utf-8").splitlines()]
        rows[0]["usable_output"] = False
        rows[0]["raw_valid"] = False
        rows[0]["fallback"] = True
        rows[0]["response_format_hash"] = "f" * 64
        tokyo_beliefs.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        invalid_report = build_smoke_report(
            results_root,
            model="fake:1b",
            model_digest=digest,
            write=False,
        )
        self.assertEqual(invalid_report["gate"], "incomplete")
        tokyo_unit = next(
            item for item in invalid_report["units"] if item["unit"] == "Tokyo"
        )
        self.assertFalse(tokyo_unit["checks"]["response_format_provenance_valid"])
        self.assertFalse(tokyo_unit["checks"]["all_outputs_valid"])

        tokyo_attempts = next(
            (results_root / "tist2015" / "Tokyo").rglob("attempts.jsonl")
        )
        with tokyo_attempts.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"query_id": "unexpected", "split": "train"}) + "\n")
        journal_report = build_smoke_report(
            results_root,
            model="fake:1b",
            model_digest=digest,
            write=False,
        )
        tokyo_unit = next(
            item for item in journal_report["units"] if item["unit"] == "Tokyo"
        )
        self.assertFalse(tokyo_unit["checks"]["attempt_ids_exact"])
        self.assertFalse(tokyo_unit["checks"]["no_dangling_ids"])
        self.assertEqual(
            tokyo_unit["journal_diagnostics"]["unexpected_attempt_ids"],
            ["unexpected"],
        )
        with self.assertRaisesRegex(SystemExit, "3"):
            main(
                [
                    "smoke-report",
                    "--results-root",
                    str(results_root),
                    "--model",
                    "fake:1b",
                    "--model-digest",
                    digest,
                    "--require-ready",
                ]
            )

    def test_official_teacher_cache_requires_pinned_model_digest(self) -> None:
        prepare(self._unit_args())
        args = self._teacher_args("official")
        args.model_digest = "unresolved"
        with self.assertRaisesRegex(ValueError, "64-hex SHA-256 --model-digest"):
            teacher_cache(args, llm_factory=lambda *_: FakeLLM())

    def test_smoke_teacher_cache_also_requires_pinned_model_digest(self) -> None:
        prepare(self._unit_args())
        args = self._teacher_args("smoke")
        args.model_digest = "unresolved"
        with self.assertRaisesRegex(ValueError, "64-hex SHA-256 --model-digest"):
            teacher_cache(args, llm_factory=lambda *_: FakeLLM())

    def test_pilot_report_rejects_duplicate_base_caches_per_unit(self) -> None:
        results_root = self.root / "duplicate-pilot-results"
        for dataset, unit in (
            ("tist2015", "Tokyo"),
            ("www2019", "Shanghai"),
            ("yjmob100k", "YJMob"),
        ):
            for suffix in ("first", "second"):
                cache = results_root / dataset / unit / "teacher" / suffix
                cache.mkdir(parents=True)
                (cache / "selection_manifest.json").write_text(
                    json.dumps(
                        {
                            "dataset": dataset,
                            "unit": unit,
                            "cache_tag": "pilot",
                        }
                    ),
                    encoding="utf-8",
                )
                (cache / "manifest.json").write_text("{}", encoding="utf-8")
        report = build_pilot_report(results_root, write=False)
        self.assertEqual(report["gate"], "incomplete")
        self.assertFalse(report["all_units_present"])
        self.assertEqual(len(report["units"]), 3)
        self.assertTrue(
            all(not item["checks"]["unique_base_cache"] for item in report["units"])
        )
        self.assertTrue(all(len(item["candidate_caches"]) == 2 for item in report["units"]))

    def test_prompt_requires_fixed_ontology_and_rejects_target_fields(self) -> None:
        query = {
            "dataset": "tist2015",
            "unit": "Tokyo",
            "user_id": "u",
            "trajectory_id": "t",
            "query_time": "2024-01-01T01:00:00Z",
            "observed_prefix": [],
            "candidate_regions": ["r0", "__OTHER__"],
        }
        prompt = build_teacher_prompt(query)
        for label in ACTIVITY_ONTOLOGY + INTENT_ONTOLOGY + DESTINATION_ONTOLOGY:
            self.assertIn(label, prompt)
        with self.assertRaisesRegex(ValueError, "forbidden target"):
            build_teacher_prompt({**query, "target_region": "r0"})

    def test_teacher_response_format_is_strict_and_parser_normalizes_weights(self) -> None:
        support = ("r0", "r1", "__OTHER__")
        response_format = build_teacher_response_format(support)
        schema = response_format["json_schema"]["schema"]
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["properties"]["next_region"]["required"]), set(support)
        )

        payload = {
            "activity": {key: 2.0 for key in ACTIVITY_ONTOLOGY},
            "intent": {key: 3.0 for key in INTENT_ONTOLOGY},
            "destination": {key: 4.0 for key in DESTINATION_ONTOLOGY},
            "next_region": {"r0": 2.0, "r1": 1.0, "__OTHER__": 1.0},
            "confidence": 0.75,
        }
        parsed = parse_teacher_response(json.dumps(payload), support)
        self.assertAlmostEqual(sum(parsed["activity"].values()), 1.0)
        self.assertAlmostEqual(sum(parsed["next_region"].values()), 1.0)
        self.assertAlmostEqual(parsed["next_region"]["r0"], 0.5)
        self.assertTrue(parsed["_normalization"]["applied"])

        payload["next_region"] = {"r0": 1.0, "r1": 1.0}
        with self.assertRaisesRegex(ValueError, "exactly the fixed support"):
            parse_teacher_response(json.dumps(payload), support)
        payload["next_region"] = {key: 0.0 for key in support}
        with self.assertRaisesRegex(ValueError, "positive finite mass"):
            parse_teacher_response(json.dumps(payload), support)

    def test_teacher_parser_rejects_duplicate_keys(self) -> None:
        raw = '{"activity":{},"activity":{},"intent":{},"destination":{},"next_region":{},"confidence":0.5}'
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            parse_teacher_response(raw, ("r0",))

    def test_lock_rejects_hand_written_or_stale_pilot_summary(self) -> None:
        results_root = self.root / "pilot-results"
        units = (("tist2015", "Tokyo"), ("www2019", "Shanghai"), ("yjmob100k", "YJMob"))
        for dataset, unit in units:
            unit_root = results_root / dataset / unit
            cache = unit_root / "teacher" / "fake" / "pilot"
            cache.mkdir(parents=True)
            query_ids = {"train": [f"train:{dataset}"], "validation": [f"validation:{dataset}"]}
            dataset_hash = hashlib.sha256(f"dataset:{dataset}".encode()).hexdigest()
            ontology_hash = hashlib.sha256(b"ontology").hexdigest()
            (cache / "selection_manifest.json").write_text(
                json.dumps(
                    {
                        "query_ids": query_ids,
                        "test_queries_read": False,
                        "dataset_hash": dataset_hash,
                        "ontology_hash": ontology_hash,
                    }
                ),
                encoding="utf-8",
            )
            belief = StructuredBelief.create(
                query=f"query:{dataset}",
                prompt="prompt",
                model_identity="fake",
                activity={key: 1.0 for key in ACTIVITY_ONTOLOGY},
                intent={key: 1.0 for key in INTENT_ONTOLOGY},
                destination={key: 1.0 for key in DESTINATION_ONTOLOGY},
                next_region={"region": 1.0},
                confidence=0.5,
            ).to_dict()
            attempt_rows, belief_rows = [], []
            for split, ids in query_ids.items():
                for query_id in ids:
                    attempt_rows.append({"query_id": query_id, "split": split})
                    belief_rows.append(
                        {
                            "query_id": query_id,
                            "split": split,
                            "raw_valid": True,
                            "transport_error": "",
                            "dataset_hash": dataset_hash,
                            "ontology_hash": ontology_hash,
                            "belief": belief,
                        }
                    )
            (cache / "attempts.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in attempt_rows), encoding="utf-8"
            )
            beliefs_path = cache / "beliefs.jsonl"
            beliefs_path.write_text(
                "".join(json.dumps(row) + "\n" for row in belief_rows), encoding="utf-8"
            )
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": dataset,
                        "unit": unit,
                        "selected_queries": 2,
                        "completed_queries": 2,
                        "raw_valid": 2,
                        "transport_errors": 0,
                        "prompt_target_leakage": 0,
                    }
                ),
                encoding="utf-8",
            )
            # This malformed test artifact must never be read by pilot report/lock.
            (unit_root / "targets").mkdir(parents=True)
            (unit_root / "targets" / "test.jsonl").write_text("{malformed", encoding="utf-8")
        incomplete = build_pilot_report(results_root, write=False)
        self.assertEqual(incomplete["gate"], "incomplete")
        protocol = self.root / "protocol.json"
        protocol.write_text(json.dumps({"version": "test"}), encoding="utf-8")
        lock_args = argparse.Namespace(results_root=str(results_root), protocol=str(protocol))
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            lock(lock_args)
        for dataset, unit in units:
            path = results_root / dataset / unit / "manifests" / "pilot_evaluation.json"
            path.parent.mkdir(parents=True)
            beliefs_path = results_root / dataset / unit / "teacher" / "fake" / "pilot" / "beliefs.jsonl"
            teacher_cache_hash = hashlib.sha256(beliefs_path.read_bytes()).hexdigest()
            path.write_text(
                json.dumps(
                    {
                        "fit_splits": ["train", "validation"],
                        "test_artifacts_read": False,
                        "teacher_cache_hash": teacher_cache_hash,
                        "teacher_activity_gain": 0.01,
                        "teacher_destination_gain": 0.02,
                        "m1_validation_nll": 2.0,
                        "m2_validation_nll": 1.9,
                        "repeat_jsd_median": 0.05,
                    }
                ),
                encoding="utf-8",
            )
        still_incomplete = build_pilot_report(results_root, write=False)
        self.assertEqual(still_incomplete["gate"], "incomplete")
        self.assertTrue(
            all(not item["pilot_evaluation_present"] for item in still_incomplete["units"])
        )
        with self.assertRaisesRegex(RuntimeError, "not ready"):
            lock(lock_args)

    def test_status_and_future_command_are_explicit(self) -> None:
        prepare(self._unit_args())
        status_args = self._unit_args("status")
        status_args.model = "fake"
        prepared = status(status_args)
        self.assertTrue(prepared["prepared"])
        self.assertEqual(prepared["stage"], "prepared")
        self.assertEqual(prepared["gate"], "prepared")
        (self.output / "targets" / "test.jsonl").unlink()
        incomplete = status(status_args)
        self.assertFalse(incomplete["prepared"])
        self.assertEqual(incomplete["stage"], "incomplete")
        self.assertIn(
            str(self.output / "targets" / "test.jsonl"),
            incomplete["missing"],
        )
        with self.assertRaisesRegex(SystemExit, "NOT_IMPLEMENTED"):
            main(["train", "--results-root", str(self.root)])


if __name__ == "__main__":
    unittest.main()
