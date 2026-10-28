import math
import unittest

from hybrid.beliefmove import (
    AnonymousCellRegionVocabulary,
    MetricSpatialRegionVocabulary,
    category_to_activity,
    category_to_destination,
    derive_intent,
    ontology_contract,
    ontology_hash,
)
from hybrid.beliefmove.regions import EARTH_RADIUS_METERS, UNKNOWN_REGION


class MetricSpatialRegionTests(unittest.TestCase):
    @staticmethod
    def point_at_x(latitude, x_meters):
        longitude = math.degrees(
            x_meters / (EARTH_RADIUS_METERS * math.cos(math.radians(latitude)))
        )
        return latitude, longitude

    def test_metric_grid_is_train_only_order_invariant_and_exact_in_local_plane(self):
        latitude = 10.0
        points = [
            self.point_at_x(latitude, 100.0),
            self.point_at_x(latitude, 300.0),
            self.point_at_x(latitude, 700.0),
        ]
        vocabulary = MetricSpatialRegionVocabulary.fit(
            points, split="train", cell_size_meters=500.0
        )
        reversed_vocabulary = MetricSpatialRegionVocabulary.fit(
            reversed(points), split="train", cell_size_meters=500.0
        )
        self.assertEqual(vocabulary.vocabulary_hash, reversed_vocabulary.vocabulary_hash)
        self.assertEqual(vocabulary.assign(*points[0]), vocabulary.assign(*points[1]))
        self.assertNotEqual(vocabulary.assign(*points[0]), vocabulary.assign(*points[2]))

        first = vocabulary.project(*self.point_at_x(latitude, 100.0))
        second = vocabulary.project(*self.point_at_x(latitude, 600.0))
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertAlmostEqual(second[0] - first[0], 500.0, places=6)
        self.assertEqual(
            vocabulary.assign(*self.point_at_x(latitude, 1_700.0)), UNKNOWN_REGION
        )
        self.assertEqual(vocabulary.assign(None, 0), UNKNOWN_REGION)
        restored = MetricSpatialRegionVocabulary.from_dict(vocabulary.to_dict())
        self.assertEqual(restored.vocabulary_hash, vocabulary.vocabulary_hash)
        with self.assertRaisesRegex(ValueError, "train"):
            MetricSpatialRegionVocabulary.fit(points, split="validation")

    def test_anonymous_cells_preserve_train_ids_and_reject_unseen(self):
        vocabulary = AnonymousCellRegionVocabulary.fit(
            ["001", "001", "cell-a"], split="train"
        )
        self.assertEqual(vocabulary.assign("001"), "cell:001")
        self.assertEqual(vocabulary.count("cell:001"), 2)
        self.assertEqual(vocabulary.assign("002"), UNKNOWN_REGION)
        restored = AnonymousCellRegionVocabulary.from_dict(vocabulary.to_dict())
        self.assertEqual(restored.vocabulary_hash, vocabulary.vocabulary_hash)
        with self.assertRaisesRegex(ValueError, "train"):
            AnonymousCellRegionVocabulary.fit(["001"], split="test")


class FrozenOntologyTests(unittest.TestCase):
    def test_category_rules_are_explicit_and_unknown_falls_back(self):
        examples = {
            "Home (private)": "residential",
            "College Cafeteria": "food-retail-service",
            "College & University": "work-education",
            "Outdoors & Recreation": "transport-outdoor",
            "Arts & Entertainment": "leisure-social",
            "entirely unseen category": "other",
            None: "other",
        }
        for category, expected in examples.items():
            with self.subTest(category=category):
                self.assertEqual(category_to_activity(category), expected)
        self.assertEqual(category_to_destination("Residence"), "home-residential")
        self.assertEqual(category_to_destination("unknown"), "other")
        self.assertEqual(len(ontology_hash()), 64)
        self.assertEqual(ontology_contract()["unknown_category"], "other")

    def test_intent_is_derived_with_frozen_precedence(self):
        pois = ["a", "b", "c"]
        regions = ["r1", "r2", "r2"]
        cases = (
            ("c", "r2", "same-poi"),
            ("d", "r2", "within-zone"),
            ("a", "r1", "return-known-poi"),
            ("d", "r1", "new-poi-known-zone"),
            ("d", "r3", "new-zone"),
            ("d", UNKNOWN_REGION, "new-zone"),
        )
        for target_poi, target_region, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    derive_intent(
                        pois,
                        regions,
                        target_poi_id=target_poi,
                        target_region_id=target_region,
                    ),
                    expected,
                )
        self.assertEqual(
            derive_intent([], [], target_poi_id="x", target_region_id="r1"),
            "new-zone",
        )
        with self.assertRaisesRegex(ValueError, "equal length"):
            derive_intent(["a"], [], target_poi_id="b", target_region_id="r2")


if __name__ == "__main__":
    unittest.main()
