import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from hybrid.direct_query_llm import collect, status, validate_ranking
from models.llm_api import LLMWrapper


class FakeWrapper:
    def __init__(self, response):
        self.response = response; self.prompts = []; self.hyperparams = {}
        self._stats = []

    def prepare_prompt(self, prompt):
        return prompt

    def get_call_stats(self):
        return list(self._stats)

    def get_response_once(self, prompt):
        self.prompts.append(prompt)
        self._stats.append({"input_tokens": 11, "output_tokens": 3,
                            "latency_seconds": .25, "physical_attempts": 1,
                            "token_source": "api", "success": True})
        return self.response


class FailingWrapper(FakeWrapper):
    def get_response_once(self, prompt):
        self.prompts.append(prompt)
        self._stats.append({"input_tokens": 11, "output_tokens": 0,
                            "latency_seconds": .25, "physical_attempts": 1,
                            "token_source": "estimated", "success": False,
                            "error": "transport failure"})
        raise ConnectionError("transport failure")


class FakeSDKClient:
    def __init__(self):
        self.max_retries = None; self.calls = 0
        self.chat = SimpleNamespace(completions=self)

    def with_options(self, max_retries):
        self.max_retries = max_retries
        return self

    def create(self, **_kwargs):
        self.calls += 1
        message = SimpleNamespace(content='{"ranking":["a"]}')
        usage = SimpleNamespace(prompt_tokens=7, completion_tokens=2)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class DirectQueryLLMTests(unittest.TestCase):
    def make_case(self, root: Path, response: str):
        ids_path = root / "candidate_ids.json"
        metadata_path = root / "candidate_metadata.json"
        logits_path = root / "test_logits.npy"
        input_path = root / "test.jsonl"
        temperature_path = root / "temperature.json"
        ids_path.write_text(json.dumps(["a", "b", "c"]), encoding="utf-8")
        metadata_path.write_text(json.dumps({
            "a": {"category": "home", "latitude": 1.0, "longitude": 2.0,
                  "address": "secret-a", "habit_score": .9},
            "b": {"category": "work", "latitude": 3.0, "longitude": 4.0,
                  "address": "secret-b", "semantic_score": .8},
        }), encoding="utf-8")
        np.save(logits_path, np.asarray([[3.0, 2.0, 1.0]]))
        rows = [
            {"_bundle": {"logits": str(logits_path), "candidate_ids": str(ids_path),
                         "candidate_metadata": str(metadata_path)}},
            {"query_id": "q1", "user_id": "u1", "city": "Tokyo", "true_id": "b",
             "_row_index": 0, "target_time": "2024-01-01 08:00:00",
             "history": [["h0", "home", "a"]], "context": [["h1", "work", "b"]]},
        ]
        input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        temperature_path.write_text(json.dumps({"temperature": {"temperature": 1.0}}), encoding="utf-8")
        args = SimpleNamespace(input=str(input_path), temperature_file=str(temperature_path),
                               output_dir=str(root / "output"), split="test",
                               model_name="qwen2:7b", model_digest="digest-1", platform="Ollama",
                               limit=1, top_k=2, top_m=1, world_mode="internal_only",
                               max_tokens=64, max_input_tokens=2000, progress_every=1)
        return args, FakeWrapper(response)

    def test_direct_one_call_uses_no_cached_evidence_and_resumes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["b","a"]}')
            metrics = collect(args, wrapper=wrapper)
            self.assertEqual(len(wrapper.prompts), 1)
            prompt = wrapper.prompts[0]
            self.assertNotIn("habit_score", prompt); self.assertNotIn("semantic_score", prompt)
            self.assertNotIn("rationale", prompt); self.assertNotIn("secret-a", prompt)
            self.assertNotIn("true_id", prompt)
            self.assertIn("quantitative_probability", prompt)
            self.assertEqual(metrics["quantitative"]["recall@1"], 0.0)
            self.assertEqual(metrics["direct-1call"]["recall@1"], 1.0)
            self.assertEqual(metrics["api_requests_per_query"], 1.0)
            second = FakeWrapper('{"ranking":["a","b"]}')
            collect(args, wrapper=second)
            self.assertEqual(second.prompts, [])

    def test_incomplete_response_falls_back_without_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["b"]}')
            metrics = collect(args, wrapper=wrapper)
            row = json.loads((root / "output/predictions.jsonl").read_text().strip())
            self.assertEqual(len(wrapper.prompts), 1)
            self.assertFalse(row["valid"]); self.assertEqual(row["direct_top_k"], ["a", "b"])
            self.assertEqual(row["true_rank"], row["quantitative_rank"])
            self.assertEqual(metrics["invalid_output_rate"], 1.0)

    def test_ranking_requires_exact_permutation(self):
        self.assertEqual(validate_ranking(["b", "a"], ["a", "b"]), (["b", "a"], True, ""))
        repaired, valid, _ = validate_ranking(["b", "invented"], ["a", "b"])
        self.assertFalse(valid); self.assertEqual(repaired, ["a", "b"])

    def test_protocol_change_refuses_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["a","b"]}')
            collect(args, wrapper=wrapper)
            args.model_digest = "different-digest"
            with self.assertRaisesRegex(ValueError, "protocol lock mismatch"):
                collect(args, wrapper=FakeWrapper('{"ranking":["a","b"]}'))

    def test_protocol_hashes_file_contents_and_selected_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["a","b"]}')
            collect(args, wrapper=wrapper)
            protocol = json.loads((root / "output/protocol.lock.json").read_text())
            self.assertEqual(len(protocol["selected_rows_sha256"]), 64)
            for key in ("jsonl", "logits", "candidate_ids", "candidate_metadata"):
                self.assertEqual(len(protocol["input"][key]["sha256"]), 64)
            np.save(root / "test_logits.npy", np.asarray([[2.0, 3.0, 1.0]]))
            with self.assertRaisesRegex(ValueError, "protocol lock mismatch"):
                collect(args, wrapper=FakeWrapper('{"ranking":["a","b"]}'))

    def test_wrapper_one_call_disables_sdk_retries(self):
        client = FakeSDKClient(); wrapper = LLMWrapper.__new__(LLMWrapper)
        wrapper.model_name = "qwen2:7b"; wrapper.api_model_name = "qwen2:7b"
        wrapper.client = client
        wrapper.hyperparams = {"temperature": 0.0, "max_tokens": 32, "max_input_tokens": 2000}
        wrapper.reset_call_stats()
        self.assertIn("ranking", wrapper.get_response_once("prompt"))
        self.assertEqual(client.max_retries, 0); self.assertEqual(client.calls, 1)
        stats = wrapper.get_call_stats()
        self.assertEqual(len(stats), 1); self.assertEqual(stats[0]["physical_attempts"], 1)
        self.assertEqual(stats[0]["token_source"], "api")

    def test_transport_failure_commits_q0_fallback_and_resume_does_not_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, _ = self.make_case(root, "")
            metrics = collect(args, wrapper=FailingWrapper(""))
            self.assertEqual(len((root / "output/requests.jsonl").read_text().splitlines()), 1)
            row = json.loads((root / "output/predictions.jsonl").read_text().strip())
            self.assertFalse(row["valid"]); self.assertFalse(row["transport_success"])
            self.assertEqual(row["api_calls"], 1)
            self.assertEqual(row["true_rank"], row["quantitative_rank"])
            self.assertIn("transport failure", row["error"])
            self.assertEqual(metrics["transport_error_rate"], 1.0)
            self.assertEqual(metrics["attempted_without_prediction"], 0)
            second = FakeWrapper('{"ranking":["b","a"]}')
            collect(args, wrapper=second)
            self.assertEqual(second.prompts, [])
            state = status(Path(args.input), Path(args.output_dir), args.limit, args.split)
            self.assertEqual(state["transport_error_rate"], 1.0)
            self.assertEqual(state["attempted_without_prediction"], 0)

    def test_crash_window_recovers_prediction_from_request_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["b","a"]}')
            collect(args, wrapper=wrapper)
            (root / "output/predictions.jsonl").write_text("", encoding="utf-8")
            interrupted = status(Path(args.input), Path(args.output_dir), args.limit, args.split)
            self.assertEqual(interrupted["attempted_without_prediction"], 1)
            self.assertEqual(interrupted["gate"], "invalid")
            second = FakeWrapper('{"ranking":["a","b"]}')
            collect(args, wrapper=second)
            self.assertEqual(second.prompts, [])
            recovered = json.loads((root / "output/predictions.jsonl").read_text().strip())
            self.assertEqual(recovered["direct_top_k"], ["b", "a"])

    def test_resume_rejects_prompt_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["a","b"]}')
            collect(args, wrapper=wrapper)
            prediction_path = root / "output/predictions.jsonl"
            row = json.loads(prediction_path.read_text().strip()); row["prompt_sha256"] = "bad"
            prediction_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "prompt_hash_mismatches"):
                collect(args, wrapper=FakeWrapper('{"ranking":["a","b"]}'))

    def test_prompt_budget_fails_before_any_api_call_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["a","b"]}')
            args.max_input_tokens = 1
            with self.assertRaisesRegex(ValueError, "prompt preflight failed"):
                collect(args, wrapper=wrapper)
            self.assertEqual(wrapper.prompts, [])
            self.assertFalse((root / "output/requests.jsonl").exists())

    def test_duplicate_request_query_id_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); args, wrapper = self.make_case(root, '{"ranking":["a","b"]}')
            collect(args, wrapper=wrapper)
            request_path = root / "output/requests.jsonl"
            line = request_path.read_text().splitlines()[0]
            request_path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate cached query_id"):
                collect(args, wrapper=FakeWrapper('{"ranking":["a","b"]}'))


if __name__ == "__main__":
    unittest.main()
