import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from hybrid.calibration import BinaryPlattCalibrator, LikelihoodRatioCalibrator, TemperatureScaler, softmax
from hybrid.bayesian_network import TrajectoryBayesianNetwork
from hybrid.evidence import AgentMoveLLMEvidenceExtractor, HeuristicEvidenceExtractor, LLMServiceUnavailable
from hybrid.experiment import run_rq_experiments
from hybrid.fusion import BayesianEvidenceFusion
from hybrid.metrics import expected_calibration_error, summarize
from hybrid.prepare_dataset import build_parser as prepare_parser, prepare
from hybrid.sample_split import sample_jsonl
from hybrid.rq_report import generate as generate_rq_report
from hybrid.enrich_osm import normalize_address
from hybrid.schemas import Candidate, Prediction, Query
from hybrid.llm_only import _parse as parse_llm_only
from hybrid.tist2015_table2_aggregate import aggregate as aggregate_tist2015_table2
from hybrid.dual_evolution import corrupt_examples, distillation_losses
from hybrid.neural_cgm import ModelConfig, _slot, build_model
from hybrid.selective_llm import SelectiveLLMPolicy
from hybrid.aggregate_runs import aggregate as aggregate_runs
from hybrid.mobility_representation import encode_trajectory, representation_hash
from hybrid.teacher_cache import ImmutableTeacherCache, canonical_hash
from hybrid.sequential_belief import SequentialBelief
from hybrid.evo_metrics import linear_cka, transition_cosine
from hybrid.beliefmove_results import aggregate as aggregate_beliefmove, load_raw, write_raw


def query(query_id, true_id, logits, city="Shanghai", backbone="test-llm"):
    return Query(
        query_id=query_id,
        user_id=f"u-{query_id}",
        city=city,
        true_id=true_id,
        backbone=backbone,
        history=[["Home", "08:00"], ["Office", "09:00"], ["Home", "18:00"]],
        context=[["Office", "17:00"]],
        target_time="18:00",
        candidates=[
            Candidate("home", logits[0], category="Home", metadata={"habit_score": 0.9, "semantic_score": 0.8}),
            Candidate("office", logits[1], category="Office", metadata={"habit_score": 0.2, "semantic_score": 0.4}),
            Candidate("food", logits[2], category="Food", metadata={"habit_score": 0.1, "semantic_score": 0.3}),
        ],
    )


class CalibrationTests(unittest.TestCase):
    def test_softmax_and_temperature(self):
        probabilities = softmax([1.0, 2.0, 3.0])
        self.assertAlmostEqual(float(probabilities.sum()), 1.0)
        scaler = TemperatureScaler().fit([[3, 1], [1, 3], [2, 1], [1, 2]], [0, 1, 0, 1])
        self.assertGreater(scaler.temperature, 0.0)

    def test_platt_constant_class(self):
        calibrator = BinaryPlattCalibrator().fit([0.1, 0.2], [0, 0])
        values = calibrator.predict([0.3, 0.4])
        self.assertTrue(np.all(values > 0.0))
        self.assertTrue(np.all(values < 1.0))

    def test_likelihood_ratio_removes_training_base_rate(self):
        calibrator = LikelihoodRatioCalibrator().fit(
            [0.1, 0.2, 0.3, 0.7, 0.8, 0.9], [0, 0, 0, 1, 1, 1]
        )
        ratios = calibrator.predict_likelihood_ratio([0.2, 0.8])
        self.assertLess(ratios[0], 1.0)
        self.assertGreater(ratios[1], 1.0)


class EvolutionAndSelectiveTests(unittest.TestCase):
    def test_time_slot_parser_handles_canonical_and_mixed_formats_without_loss(self):
        values = pd.Series(["2012-04-14 16:45:31", "2012-04-15T10:45:53Z", None])
        self.assertEqual(_slot(values), [33, 21, 0])

    def test_order_corruption_preserves_targets_and_alignment(self):
        example = [([1, 2, 3], [10, 20, 30], 4, 31, 9)]
        reversed_rows = corrupt_examples(example, "reverse", 42)
        self.assertEqual(reversed_rows[0], ([3, 2, 1], [30, 20, 10], 4, 31, 9))
        self.assertEqual(corrupt_examples(example, "correct", 42), example)

    def test_selective_llm_entropy_and_margin(self):
        policy = SelectiveLLMPolicy(entropy_threshold=0.5, margin_threshold=0.2)
        self.assertFalse(policy.decide([0.9, 0.1])["call_llm"])
        self.assertTrue(policy.decide([0.51, 0.49])["call_llm"])

    def test_neural_cgm_exposes_depth_and_temporal_states(self):
        import torch
        model = build_model(ModelConfig(num_pois=5, num_users=2, hidden_dim=8))
        output = model(
            torch.tensor([[1, 2, 0], [2, 3, 4]]), torch.tensor([[1, 2, 0], [3, 4, 5]]),
            torch.tensor([2, 3]), torch.tensor([0, 1]), torch.tensor([3, 6]), return_states=True,
        )
        self.assertEqual(tuple(output["logits"].shape), (2, 5))
        self.assertEqual(len(output["depth_states"]), 2)
        self.assertEqual(tuple(output["temporal_states"].shape), (2, 3, 8))
        projections = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
        losses = distillation_losses(output, output, torch.tensor([1, 2]), projections, 2.0)
        self.assertAlmostEqual(float(losses["kd"].detach()), 0.0, places=5)
        self.assertAlmostEqual(float(losses["temporal"].detach()), 0.0, places=5)

    def test_run_aggregator_indexes_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); target = root / "city" / "full"; target.mkdir(parents=True)
            (target / "metrics.json").write_text(json.dumps({"acc@1": 0.5}))
            payload = aggregate_runs(root)
            self.assertEqual(payload["completed_metric_files"], 1)
            self.assertEqual(payload["runs"][0]["metrics"]["acc@1"], 0.5)

    def test_representation_is_deterministic_and_uses_past_frequency_only(self):
        events = [
            {"poi_id": "home", "timestamp": "2026-01-05T08:00:00", "heading": 90},
            {"poi_id": "home", "timestamp": "2026-01-05T09:00:00", "heading": 90},
        ]
        first = encode_trajectory(events); second = encode_trajectory(events)
        self.assertEqual(first, second); self.assertEqual(representation_hash(first), representation_hash(second))
        self.assertEqual(first[0].historical_frequency, 0.0)
        self.assertEqual(first[1].historical_frequency, 1.0)

    def test_teacher_cache_is_immutable_and_resumable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.jsonl"
            cache = ImmutableTeacherCache(path, "llm", "v1")
            cache.put("q1", {"intent": {"home": 0.8}}, canonical_hash({"query": 1}))
            cache.put("q1", {"intent": {"home": 0.8}}, canonical_hash({"query": 1}))
            self.assertEqual(len(path.read_text().splitlines()), 1)
            with self.assertRaisesRegex(ValueError, "immutable"):
                cache.put("q1", {"intent": {"home": 0.2}}, canonical_hash({"query": 1}))

    def test_sequential_belief_is_normalized_and_history_changes_state(self):
        belief = SequentialBelief([0.5, 0.5], [[0.9, 0.1], [0.2, 0.8]])
        first = belief.step([0.8, 0.2]); second = belief.step([0.8, 0.2])
        self.assertAlmostEqual(float(first.sum()), 1.0); self.assertAlmostEqual(float(second.sum()), 1.0)
        self.assertFalse(np.allclose(first, second))

    def test_evolution_metrics(self):
        values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.assertAlmostEqual(linear_cka(values, values), 1.0)
        temporal = np.array([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]])
        self.assertAlmostEqual(transition_cosine(temporal, temporal), 1.0)

    def test_raw_result_schema_and_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); repository = Path(__file__).resolve().parents[3]
            for seed, value in [(42, 0.4), (43, 0.6)]:
                write_raw(root / f"seed-{seed}.json", "RQ4", "E5-dual", seed, "toy", "config.json", {"acc1": value}, repository)
            summary = aggregate_beliefmove(load_raw(root))
            self.assertEqual(summary["raw_runs"], 2)
            self.assertAlmostEqual(summary["groups"][0]["metrics"]["acc1"]["mean"], 0.5)
            self.assertEqual(len(summary["groups"][0]["metrics"]["acc1"]["bootstrap_ci95"]), 2)


class FusionAndMetricTests(unittest.TestCase):
    def test_explicit_bbn_structure_and_exact_inference(self):
        network = TrajectoryBayesianNetwork(["a", "b"])
        structure = network.structure()
        self.assertEqual(structure["edges"], [
            {"parent": "L", "child": "H"},
            {"parent": "L", "child": "S"},
        ])
        posterior, terms = (
            network.set_prior([0.6, 0.4])
            .observe("H", [0.8, 0.3])
            .observe("S", [0.7, 0.6])
            .infer()
        )
        expected_a = 0.6 * 0.8 * 0.7
        expected_b = 0.4 * 0.3 * 0.6
        self.assertAlmostEqual(posterior[0], expected_a / (expected_a + expected_b))
        self.assertAlmostEqual(np.exp(terms["a"]["log_posterior"]), posterior[0])

    def test_bbn_rejects_missing_evidence(self):
        with self.assertRaisesRegex(RuntimeError, "missing observed evidence"):
            TrajectoryBayesianNetwork(["a", "b"]).set_prior([0.5, 0.5]).observe("H", [0.5, 0.5]).infer()

    def test_fusion_is_normalized_and_decomposable(self):
        posterior, terms = BayesianEvidenceFusion().fuse(
            ["a", "b"], [0.6, 0.4], [0.8, 0.3], [0.7, 0.6]
        )
        self.assertAlmostEqual(float(posterior.sum()), 1.0)
        for candidate_id, probability in zip(["a", "b"], posterior):
            self.assertAlmostEqual(math.exp(terms[candidate_id]["log_posterior"]), probability)

    def test_ece_perfect(self):
        self.assertAlmostEqual(expected_calibration_error([1.0, 1.0], [1, 1]), 0.0)


class IORegressionTests(unittest.TestCase):
    def test_tist2015_table2_aggregate_marks_partial_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hybrid_root = root / "hybrid"
            llm_root = root / "llm"
            metrics = {"queries": 200, "acc@1": 0.1, "acc@5": 0.2, "acc@10": 0.3, "mrr": 0.15}
            for variant in ("full", "stage1_only"):
                target = hybrid_root / "Tokyo" / variant
                target.mkdir(parents=True)
                (target / "metrics.json").write_text(json.dumps(metrics))
            llm_city = llm_root / "Tokyo"
            llm_city.mkdir(parents=True)
            (llm_city / "metrics.json").write_text(json.dumps({
                "queries": 200, "acc@1": 0.05, "acc@5": 0.2, "mrr": 0.1,
            }))
            (llm_city / "protocol.json").write_text(json.dumps({
                "model": "qwen2:7b", "sample_mode": "matched-test-prefix", "requested_limit": 200,
            }))
            summary = aggregate_tist2015_table2(
                hybrid_root, llm_root, root / "out", "qwen2:7b", 200
            )
            self.assertFalse(summary["publication_ready"])
            self.assertEqual(summary["hybrid_no_osm"]["completed_cities"], ["Tokyo"])
            self.assertIn("SanFrancisco", summary["llm_zs"]["missing_or_incompatible_cities"])
            self.assertIn("\\dagger", (root / "out" / "tist2015_table2_rows.tex").read_text())

    def test_llm_only_parser_accepts_object_and_direct_list(self):
        self.assertEqual(parse_llm_only('{"prediction":["1","2"]}'), ["1", "2"])
        self.assertEqual(parse_llm_only('["1","2"]'), ["1", "2"])

    def test_hash_sample_is_reproducible_and_order_independent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"query_id": f"q{i}"} for i in range(10)]
            first_source = root / "first.jsonl"
            second_source = root / "second.jsonl"
            first_source.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            second_source.write_text("\n".join(json.dumps(row) for row in reversed(rows)) + "\n")
            first = sample_jsonl(first_source, root / "first-out.jsonl", 0.5, 42)
            second = sample_jsonl(second_source, root / "second-out.jsonl", 0.5, 42)
            self.assertEqual(first["selected_queries"], 5)
            self.assertEqual(first["query_ids_sha256"], second["query_ids_sha256"])

    def test_missing_jsonl_error_is_actionable(self):
        missing = "data/hybrid/validation.jsonl"
        with self.assertRaises(FileNotFoundError) as ctx:
            from hybrid.io import read_jsonl
            list(read_jsonl(missing))
        self.assertIn("not found", str(ctx.exception).lower())
        self.assertIn(missing, str(ctx.exception))
        self.assertIn("--validation", str(ctx.exception))


class ExperimentSmokeTest(unittest.TestCase):
    def test_llm_evidence_parser_accepts_wrapped_and_direct_lists(self):
        row = {"candidate_id": "home", "habit_score": 0.8, "semantic_score": 0.7}
        self.assertEqual(
            AgentMoveLLMEvidenceExtractor._parse(json.dumps({"evidence": [row]})),
            [row],
        )
        self.assertEqual(AgentMoveLLMEvidenceExtractor._parse(json.dumps([row])), [row])
        self.assertEqual(
            AgentMoveLLMEvidenceExtractor._parse(f"```json\n{json.dumps([row])}\n```"),
            [row],
        )

    def test_endpoint_failure_is_not_converted_to_neutral_evidence(self):
        class FailingWrapper:
            def get_response(self, prompt):
                raise RuntimeError("connection refused")

        extractor = object.__new__(AgentMoveLLMEvidenceExtractor)
        extractor.wrapper = FailingWrapper()
        sample = query("offline", "home", [2.0, 1.0, 0.0])
        with self.assertRaisesRegex(LLMServiceUnavailable, "connection refused"):
            extractor._extract_batch(sample, sample.candidates[:1], [])

    def test_missing_llm_evidence_becomes_explicit_neutral_record(self):
        extractor = object.__new__(AgentMoveLLMEvidenceExtractor)
        extractor.batch_size = 2
        extractor.retries = 1
        extractor.missing_policy = "neutral"
        extractor._extract_batch = lambda query, candidates, memory: []
        sample = query("missing", "home", [2.0, 1.0, 0.0])
        evidence = extractor.extract(sample, sample.candidates[:2], [])
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all(not item.valid for item in evidence))
        self.assertTrue(all(item.habit_score == 0.5 for item in evidence))

    def test_osm_address_normalization(self):
        result = normalize_address({
            "category": "amenity", "type": "university", "name": "Example University",
            "address": {"state": "Hanoi", "suburb": "Cau Giay", "road": "Main Road"},
        })
        self.assertEqual(result["admin"], "Hanoi")
        self.assertEqual(result["subdistrict"], "Cau Giay")
        self.assertEqual(result["poi"], "Example University")
    def test_all_rq_artifacts_are_created(self):
        validation = [
            query("v1", "home", [2.0, 1.0, 0.0]),
            query("v2", "office", [0.5, 2.0, 0.0]),
            query("v3", "home", [1.5, 1.0, 0.2]),
        ]
        test = [
            query("t1", "home", [1.0, 1.2, 0.0]),
            query("t2", "office", [0.4, 1.8, 0.1], city="Tokyo"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            summaries = run_rq_experiments(
                validation, test, HeuristicEvidenceExtractor(), directory, top_k=3, top_m=2
            )
            self.assertIn("full", summaries)
            for name in [
                "rq1_main_results.json",
                "rq2_ablation.json",
                "rq3_efficiency.json",
                "rq4_calibration_and_generalization.json",
                "bbn_structure.json",
            ]:
                self.assertTrue((Path(directory) / name).exists(), name)
            manifest = {
                "validation": {"selected_queries": 3, "source_queries": 6, "seed": 42,
                               "selection": "test", "query_ids_sha256": "validation"},
                "test": {"selected_queries": 2, "source_queries": 4, "seed": 42,
                         "selection": "test", "query_ids_sha256": "test"},
            }
            manifest_path = Path(directory) / "sample_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            report = generate_rq_report(Path(directory), manifest_path)
            self.assertTrue(report.exists())
            self.assertIn("RQ1", report.read_text())
            self.assertTrue((Path(directory) / "rq4_reliability_bins.json").exists())
            predictions = (Path(directory) / "full" / "predictions.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(predictions), 2)
            self.assertEqual(json.loads(predictions[0])["variant"], "full")

    def test_dataset_preparation_exports_all_required_files(self):
        rows = []
        for user_index in range(2):
            for trajectory_index in range(10):
                for point_index in range(4):
                    rows.append({
                        "city": "Shanghai",
                        "user_id": f"u{user_index}",
                        "traj_id": trajectory_index,
                        "venue_id": ["a", "b", "c", "a"][point_index],
                        "utc_time": f"2024-01-{trajectory_index + 1:02d} {8 + point_index:02d}:00:00",
                        "latitude": 31.2 + point_index / 100,
                        "longitude": 121.4 + point_index / 100,
                        "venue_category_name": ["Home", "Work", "Food", "Home"][point_index],
                    })
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "shanghai.csv"
            output = Path(directory) / "output"
            pd.DataFrame(rows).to_csv(source, index=False)
            args = prepare_parser().parse_args([
                "--dataset", "isp", "--input", str(source), "--city", "Shanghai",
                "--output-dir", str(output),
            ])
            stats = prepare(args)
            self.assertGreater(stats["validation_queries"], 0)
            for name in [
                "validation_logits.npy", "test_logits.npy", "validation_metadata.jsonl",
                "test_metadata.jsonl", "candidate_ids.json", "candidate_metadata.json", "validation.jsonl", "test.jsonl",
                "getnext/train.csv", "getnext/val.csv", "getnext/test.csv",
            ]:
                self.assertTrue((output / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
