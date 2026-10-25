from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from phase2.aggregate_direct_query_llm import (
    ROUTING_POLICIES,
    _expected_direct_metrics,
    _expected_routing_metrics,
    _sha256_file,
    _sha256_json,
    _validate_direct_unit,
    _validate_routing_unit,
    clustered_test,
    collect,
)


class DirectQueryAggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.folder = self.root / "Fixture/direct-query-llm/qwen2-7b/limit-2"
        self.digest = "sha256:model-digest"
        self.seeds = [42, 43]
        self._build_direct_unit()
        self._build_routing()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _write_jsonl(path: Path, rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                        encoding="utf-8")

    def _input(self, split: str) -> dict:
        return {
            "jsonl": {"path": f"/{split}.jsonl", "sha256": f"{split}-jsonl-hash"},
            "logits": {"path": f"/{split}.npy", "sha256": f"{split}-logits-hash"},
            "candidate_ids": {"path": "/candidate_ids.json", "sha256": "candidate-file-hash"},
            "candidate_metadata": {"path": "/metadata.json", "sha256": "metadata-file-hash"},
            "selected_rows_sha256": f"{split}-selected-rows-hash",
        }

    def _rows(self, split: str) -> list[dict]:
        query_ids = [f"{split}-q1", f"{split}-q2"]
        definitions = (("a", 1, 2), ("b", 2, 1))
        rows = []
        for index, (true_id, quantitative_rank, direct_rank) in enumerate(definitions):
            prompt_sha = f"prompt-{split}-{index}"
            rows.append({
                "protocol": "direct-query-llm-v1",
                "query_id": query_ids[index],
                "user_id": f"user-{index + 1}",
                "city": "Fixture",
                "true_id": true_id,
                "stage1_top_k": ["a", "b"],
                "direct_top_k": ["b", "a"],
                "quantitative_rank": quantitative_rank,
                "true_rank": direct_rank,
                "true_in_top_k": True,
                "valid": True,
                "transport_success": True,
                "error": "",
                "api_calls": 1,
                "physical_attempts": 1,
                "input_tokens": 100 + index,
                "output_tokens": 10 + index,
                "token_source": "api",
                "latency_seconds": 1.0 + index,
                "prompt_sha256": prompt_sha,
                "original_prompt_sha256": prompt_sha,
                "sent_prompt_sha256": prompt_sha,
                "original_prompt_chars": 400 + index,
                "sent_prompt_chars": 400 + index,
                "original_prompt_estimated_tokens": 100 + index,
                "sent_prompt_estimated_tokens": 100 + index,
                "truncated": False,
                "response_sha256": f"response-{split}-{index}",
                "candidate_ids_sha256": _sha256_json(["a", "b"]),
                "model": "qwen2:7b",
                "model_digest": self.digest,
                "world_mode": "internal_only",
                "temperature": 1.25,
            })
        return rows

    def _protocol(self, split: str, rows: list[dict]) -> dict:
        contract = {
            "prompt_version": "direct-query-llm-prompt-v1",
            "top_k": 2,
            "top_m": 5,
            "world_mode": "internal_only",
            "uses_quantitative_probabilities": True,
            "uses_structured_llm_evidence": False,
            "uses_test_labels_in_prompt": False,
        }
        inputs = self._input(split)
        return {
            "protocol": "direct-query-llm-v1",
            "split": split,
            "limit": 2,
            "queries": 2,
            "query_ids_sha256": _sha256_json([row["query_id"] for row in rows]),
            "selected_rows_sha256": inputs["selected_rows_sha256"],
            "candidate_ids_sha256": _sha256_json(["a", "b"]),
            "prompt_contract": contract,
            "prompt_contract_sha256": _sha256_json(contract),
            "temperature": 1.25,
            "model": "qwen2:7b",
            "model_digest": self.digest,
            "platform": "Ollama",
            "world_mode": "internal_only",
            "temperature_parameter": 0.0,
            "max_tokens": 256,
            "max_input_tokens": 8192,
            "physical_requests_per_uncached_query": 1,
            "automatic_retries": False,
            "input": inputs,
        }

    def _build_split(self, split: str) -> tuple[list[dict], dict]:
        output = self.folder / "always-cache" / split
        rows = self._rows(split)
        protocol = self._protocol(split, rows)
        requests = []
        for row in rows:
            requests.append({
                "query_id": row["query_id"],
                "split": split,
                "api_calls": 1,
                "physical_attempts": 1,
                "success": True,
                "status": "success",
                "prompt_sha256": row["prompt_sha256"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "latency_seconds": row["latency_seconds"],
                "prediction": row,
            })
        metrics = _expected_direct_metrics(rows, split, 2)
        metrics.update({"prompts_truncated": False})
        self._write_json(output / "protocol.lock.json", protocol)
        self._write_jsonl(output / "predictions.jsonl", rows)
        self._write_jsonl(output / "requests.jsonl", requests)
        self._write_json(output / "metrics.json", metrics)
        np.savez_compressed(
            output / "predictions.npz",
            query_id=np.asarray([row["query_id"] for row in rows]),
            user_id=np.asarray([row["user_id"] for row in rows]),
            quantitative_ranks=np.asarray([row["quantitative_rank"] for row in rows], dtype=np.int32),
            direct_ranks=np.asarray([row["true_rank"] for row in rows], dtype=np.int32),
            ranks=np.asarray([row["true_rank"] for row in rows], dtype=np.int32),
        )
        return rows, protocol

    def _build_direct_unit(self) -> None:
        validation_rows, validation_protocol = self._build_split("validation")
        self.test_rows, _ = self._build_split("test")
        temperature = {
            "protocol": "direct-query-llm-v1",
            "fit_split": "validation",
            "limit": 2,
            "queries": 2,
            "query_ids_sha256": validation_protocol["query_ids_sha256"],
            "selected_rows_sha256": validation_protocol["selected_rows_sha256"],
            "candidate_ids_sha256": validation_protocol["candidate_ids_sha256"],
            "input": validation_protocol["input"],
            "temperature": {"temperature": 1.25},
        }
        self._write_json(self.folder / "stage1_temperature.json", temperature)

    def _policy_arrays(self, seed: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        quantitative = np.asarray([1, 2], dtype=np.int32)
        direct = np.asarray([2, 1], dtype=np.int32)
        random_called = (np.asarray([1, 0], dtype=np.int8) if seed == 42
                         else np.asarray([0, 1], dtype=np.int8))
        return {
            "never": (quantitative, np.asarray([0, 0], dtype=np.int8)),
            "always": (direct, np.asarray([1, 1], dtype=np.int8)),
            "entropy": (np.asarray([2, 2], dtype=np.int32), np.asarray([1, 0], dtype=np.int8)),
            "margin": (np.asarray([1, 1], dtype=np.int32), np.asarray([0, 1], dtype=np.int8)),
            "random-budget-matched": (np.where(random_called, direct, quantitative), random_called),
        }

    def _build_routing(self) -> None:
        query_ids = np.asarray([row["query_id"] for row in self.test_rows])
        user_ids = np.asarray([row["user_id"] for row in self.test_rows])
        validation_protocol = json.loads(
            (self.folder / "always-cache/validation/protocol.lock.json").read_text())
        test_protocol = json.loads((self.folder / "always-cache/test/protocol.lock.json").read_text())
        provenance = {
            "routing_config_sha256": "routing-config-hash",
            "calibration_sha256": _sha256_file(self.folder / "stage1_temperature.json"),
            "validation_input_sha256": validation_protocol["input"]["jsonl"]["sha256"],
            "test_input_sha256": test_protocol["input"]["jsonl"]["sha256"],
            "validation_always_sha256": _sha256_file(
                self.folder / "always-cache/validation/predictions.jsonl"),
            "test_always_sha256": _sha256_file(
                self.folder / "always-cache/test/predictions.jsonl"),
        }
        for seed in self.seeds:
            output = self.folder / f"routing/seed-{seed}"
            output.mkdir(parents=True, exist_ok=True)
            policy_arrays = self._policy_arrays(seed)
            metrics = {}
            for policy in ROUTING_POLICIES:
                ranks, called = policy_arrays[policy]
                metrics[policy] = _expected_routing_metrics(ranks, called, self.test_rows)
                np.savez_compressed(output / f"{policy}.test.predictions.npz",
                                    query_id=query_ids, user_id=user_ids,
                                    ranks=ranks, called_llm=called)
            np.savez_compressed(output / "routing_diagnostics.test.npz",
                                query_id=query_ids,
                                stage_ranks=np.asarray([1, 2], dtype=np.int32),
                                llm_ranks=np.asarray([2, 1], dtype=np.int32),
                                entropy=np.asarray([2.0, 0.5]),
                                margin=np.asarray([0.1, 0.9]),
                                oracle_called=np.asarray([0, 1], dtype=np.int8))
            payload = {
                "rq": "RQ8",
                "seed": seed,
                "fit_split": "validation",
                "evaluation_split": "test",
                "limit": 2,
                "target_call_rate": 0.5,
                "routing_mode": "counterfactual_replay_from_always_response_bank",
                "physical_requests_during_routing": 0,
                "reported_request_cost": "simulated_deployment_budget",
                "provenance": provenance,
                "selected_thresholds": {"entropy": 1.0, "margin": 0.1},
                "validation_selection": {"entropy": {"threshold": 1.0, "call_rate": 0.5}},
                "budget_sweep": {"0.5": {"entropy": {"selected_query_rate": 0.5}}},
                "random_matched_to": "entropy test call rate",
                "metrics": metrics,
            }
            self._write_json(output / "rq8.metrics.json", payload)

    def _validated_direct(self) -> dict:
        return _validate_direct_unit(self.folder, 2, "qwen2-7b")

    def test_ready_synthetic_gate_passes(self) -> None:
        spec = {"key": "fixture", "label": "Fixture", "root": self.root,
                "units": ["Fixture"]}
        with patch("phase2.aggregate_direct_query_llm.dataset_specs", return_value=[spec]):
            payload = collect(self.root, "qwen2-7b", 2, ["Fixture"], self.seeds, 1000)
        self.assertEqual(payload["gate"], "ready-bounded")
        self.assertEqual(payload["missing"], [])
        self.assertEqual(payload["transport_errors"], 0)
        self.assertEqual(payload["model_digests"], [self.digest])

    def test_missing_request_journal_fails(self) -> None:
        (self.folder / "always-cache/test/requests.jsonl").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "requests.jsonl"):
            self._validated_direct()

    def test_misaligned_request_journal_fails(self) -> None:
        path = self.folder / "always-cache/test/requests.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["query_id"] = "wrong-query"
        self._write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "journal/prediction query set or order mismatch"):
            self._validated_direct()

    def test_transport_error_is_counted_without_retry_or_gate_failure(self) -> None:
        root = self.folder / "always-cache/test"
        predictions = [json.loads(line) for line in
                       (root / "predictions.jsonl").read_text().splitlines()]
        failed = predictions[0]
        failed["direct_top_k"] = list(failed["stage1_top_k"])
        failed["true_rank"] = failed["quantitative_rank"]
        failed["valid"] = False
        failed["transport_success"] = False
        failed["error"] = "TimeoutError: timed out"
        failed["response_sha256"] = None
        self._write_jsonl(root / "predictions.jsonl", predictions)

        requests = [json.loads(line) for line in (root / "requests.jsonl").read_text().splitlines()]
        requests[0]["status"] = "transport_error"
        requests[0]["success"] = False
        requests[0]["error"] = failed["error"]
        requests[0]["prediction"] = failed
        self._write_jsonl(root / "requests.jsonl", requests)
        metrics = _expected_direct_metrics(predictions, "test", 2)
        metrics["prompts_truncated"] = False
        self._write_json(root / "metrics.json", metrics)
        direct_ranks = np.asarray([row["true_rank"] for row in predictions], dtype=np.int32)
        np.savez_compressed(root / "predictions.npz",
                            query_id=np.asarray([row["query_id"] for row in predictions]),
                            user_id=np.asarray([row["user_id"] for row in predictions]),
                            quantitative_ranks=np.asarray(
                                [row["quantitative_rank"] for row in predictions], dtype=np.int32),
                            direct_ranks=direct_ranks, ranks=direct_ranks)

        validated = self._validated_direct()
        self.assertEqual(validated["test"]["transport_errors"], 1)
        self.assertEqual(validated["test"]["transport_error_query_ids"], ["test-q1"])
        self.assertEqual(validated["test"]["metrics"]["transport_error_rate"], 0.5)

    def test_wrong_prompt_protocol_flags_fail(self) -> None:
        path = self.folder / "always-cache/test/protocol.lock.json"
        protocol = json.loads(path.read_text())
        protocol["prompt_contract"]["uses_structured_llm_evidence"] = True
        protocol["prompt_contract_sha256"] = _sha256_json(protocol["prompt_contract"])
        self._write_json(path, protocol)
        with self.assertRaisesRegex(ValueError, "illegal Direct-1Call prompt inputs"):
            self._validated_direct()

    def test_routing_direct_alignment_fails(self) -> None:
        direct = self._validated_direct()
        path = self.folder / "routing/seed-42/always.test.predictions.npz"
        arrays = dict(np.load(path, allow_pickle=False))
        arrays["user_id"] = np.asarray(["wrong-user", "user-2"])
        np.savez_compressed(path, **arrays)
        with self.assertRaisesRegex(ValueError, "routing/direct alignment mismatch"):
            _validate_routing_unit(self.folder, self.seeds, direct, 2)

    def test_routing_config_identity_fails(self) -> None:
        direct = self._validated_direct()
        path = self.folder / "routing/seed-43/rq8.metrics.json"
        payload = json.loads(path.read_text())
        payload["provenance"]["routing_config_sha256"] = "different-config"
        self._write_json(path, payload)
        with self.assertRaisesRegex(ValueError, "configuration identity differs"):
            _validate_routing_unit(self.folder, self.seeds, direct, 2)

    def test_clustered_test_basic_behavior(self) -> None:
        values = np.ones((20, 2), dtype=float)
        users = np.repeat([f"user-{index}" for index in range(10)], 2)
        effect, interval, pvalue = clustered_test([(values, users)], 2000, 123)
        np.testing.assert_allclose(effect, [1.0, 1.0])
        np.testing.assert_allclose(interval, [[1.0, 1.0], [1.0, 1.0]])
        self.assertTrue(np.all(pvalue < 0.05))


if __name__ == "__main__":
    unittest.main()
