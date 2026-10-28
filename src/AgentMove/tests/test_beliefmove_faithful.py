import copy
import unittest

import numpy as np

from hybrid.beliefmove import (
    BlendedCPT,
    DataCPT,
    RegionAwareDecoder,
    SequentialBeliefState,
    SpatialRegionVocabulary,
    StructuredBelief,
    TeacherSoftCountCPT,
    fit_transition_matrix,
    predictive_entropy,
    top2_margin,
)


class StructuredBeliefTests(unittest.TestCase):
    def belief(self):
        return StructuredBelief.create(
            query="user=1|prefix=a,b",
            prompt="Return the fixed ontology as JSON",
            model_identity="qwen2.5:7b@digest",
            activity={"work": 2, "other": 1},
            intent={"routine": 4, "explore": 1},
            destination={"poi-a": 1, "poi-b": 3},
            next_region={"region-a": 2, "region-b": 2},
            confidence=0.75,
        )

    def test_round_trip_is_normalized_and_hash_verified(self):
        belief = self.belief()
        self.assertAlmostEqual(sum(belief.destination.values()), 1.0)
        restored = StructuredBelief.from_dict(belief.to_dict())
        self.assertEqual(restored.canonical_json(), belief.canonical_json())

    def test_strict_parser_rejects_tamper_extra_and_unnormalized_payload(self):
        payload = self.belief().to_dict()
        tampered = copy.deepcopy(payload)
        tampered["confidence"] = 0.2
        with self.assertRaisesRegex(ValueError, "belief_hash"):
            StructuredBelief.from_dict(tampered)
        extra = copy.deepcopy(payload)
        extra["free_text"] = "not allowed"
        with self.assertRaisesRegex(ValueError, "fields mismatch"):
            StructuredBelief.from_dict(extra)
        unnormalized = copy.deepcopy(payload)
        unnormalized["intent"] = {"routine": 4.0, "explore": 1.0}
        with self.assertRaisesRegex(ValueError, "sum to one"):
            StructuredBelief.from_dict(unnormalized)


class SpatialVocabularyTests(unittest.TestCase):
    def test_train_only_fit_is_deterministic_and_unseen_cells_are_unknown(self):
        points = [(10.001, 106.001), (10.002, 106.002), (10.021, 106.021), (None, 0)]
        first = SpatialRegionVocabulary.fit(points, split="train", grid_size_degrees=0.01)
        second = SpatialRegionVocabulary.fit(reversed(points), split="train", grid_size_degrees=0.01)
        self.assertEqual(first.vocabulary_hash, second.vocabulary_hash)
        self.assertNotEqual(first.assign(10.001, 106.001), first.unknown_region)
        self.assertEqual(first.assign(11.0, 107.0), first.unknown_region)
        self.assertEqual(first.assign(None, 107.0), first.unknown_region)
        restored = SpatialRegionVocabulary.from_dict(first.to_dict())
        self.assertEqual(restored.vocabulary_hash, first.vocabulary_hash)
        with self.assertRaisesRegex(ValueError, "train"):
            SpatialRegionVocabulary.fit(points, split="validation")


class CPTTests(unittest.TestCase):
    def test_teacher_soft_counts_blend_and_alpha_is_validation_selected(self):
        labels = ("a", "b")
        data = DataCPT(labels, smoothing=0.01).fit(
            [("x", "a"), ("x", "a"), ("y", "a")], split="train"
        )
        teacher = TeacherSoftCountCPT(labels, smoothing=0.01).fit(
            [("x", {"a": 0.0, "b": 1.0}), ("y", {"a": 0.0, "b": 1.0})], split="train"
        )
        blended = BlendedCPT(data, teacher)
        selection = blended.select_alpha(
            [("x", "b"), ("y", "b")], split="validation", grid=(0.0, 0.5, 1.0)
        )
        self.assertEqual(selection.alpha, 0.0)
        self.assertGreater(blended.distribution("x")["b"], blended.distribution("x")["a"])
        with self.assertRaisesRegex(ValueError, "validation"):
            blended.select_alpha([("x", "b")], split="test")
        with self.assertRaisesRegex(ValueError, "train"):
            data.fit([("x", "a")], split="test")


class SequentialBeliefTests(unittest.TestCase):
    def test_predict_update_and_reset(self):
        transition = fit_transition_matrix(
            [["a", "b", "b"], ["a", "b"]], ("a", "b"), split="train", smoothing=0.1
        )
        state = SequentialBeliefState(("a", "b"), transition, initial={"a": 1.0, "b": 0.0})
        predicted = state.predict()
        self.assertGreater(predicted[1], predicted[0])
        posterior = state.update({"a": 0.9, "b": 0.1})
        self.assertGreater(posterior[0], predicted[0])
        np.testing.assert_allclose(state.reset(), [1.0, 0.0])
        with self.assertRaisesRegex(ValueError, "train"):
            fit_transition_matrix([["a", "b"]], ("a", "b"), split="validation")


class UncertaintyAndDecoderTests(unittest.TestCase):
    def test_entropy_and_margin_have_expected_order(self):
        certain = [0.9, 0.1]
        uncertain = [0.5, 0.5]
        self.assertGreater(predictive_entropy(uncertain), predictive_entropy(certain))
        self.assertLess(top2_margin(uncertain), top2_margin(certain))
        with self.assertRaisesRegex(ValueError, "sum to one"):
            predictive_entropy([0.8, 0.8])

    def test_region_decoder_fuses_without_mutating_frozen_quantitative_prior(self):
        vocabulary = SpatialRegionVocabulary.fit(
            [(10.001, 106.001), (10.002, 106.002), (10.021, 106.021)],
            split="train",
            grid_size_degrees=0.01,
        )
        region_a = vocabulary.assign(10.001, 106.001)
        region_b = vocabulary.assign(10.021, 106.021)
        decoder = RegionAwareDecoder(
            vocabulary,
            {"poi-a": region_a, "poi-b": region_b},
            poi_coordinates={"poi-a": (10.001, 106.001), "poi-b": (10.021, 106.021)},
        )
        quantitative = {"poi-a": 0.7, "poi-b": 0.3}
        frozen_copy = quantitative.copy()
        decoded = decoder.decode({region_a: 0.05, region_b: 0.95}, quantitative)
        self.assertEqual(decoded.poi_id, "poi-b")
        self.assertEqual(quantitative, frozen_copy)
        self.assertAlmostEqual(sum(decoded.poi_probabilities.values()), 1.0)
        self.assertIsNotNone(decoded.centroid)
        no_region_effect = decoder.decode({region_a: 0.05, region_b: 0.95}, quantitative, region_strength=0.0)
        self.assertEqual(no_region_effect.poi_id, "poi-a")


if __name__ == "__main__":
    unittest.main()
