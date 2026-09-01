import unittest

import pandas as pd

from hybrid.rq13_robustness import perturb_examples, position_mapping


class RQ13RobustnessTests(unittest.TestCase):
    def setUp(self):
        self.examples = [([0, 1, 2, 3], [1, 2, 3, 4], 0, 5, 4),
                         ([1, 2, 3], [2, 3, 4], 1, 6, 0)]

    def test_gps_dropout_is_deterministic_and_preserves_last_observation_and_label(self):
        left, rate = perturb_examples(self.examples, "gps-drop-50", 42, 2)
        right, _ = perturb_examples(self.examples, "gps-drop-50", 42, 2)
        self.assertEqual(left, right); self.assertGreater(rate, 0)
        for original, changed in zip(self.examples, left):
            self.assertEqual(changed[0][-1], original[0][-1]); self.assertEqual(changed[-1], original[-1])

    def test_context_corruption_keeps_target_label(self):
        missing, _ = perturb_examples(self.examples, "context-missing", 42, 2)
        wrong, _ = perturb_examples(self.examples, "context-wrong", 42, 2)
        self.assertEqual([row[-1] for row in missing], [row[-1] for row in self.examples])
        self.assertTrue(all(row[2] == 2 and row[3] == 0 for row in missing))
        self.assertEqual([row[2] for row in wrong], [1, 0])

    def test_position_mapping_returns_candidate_ids(self):
        frame = pd.DataFrame({"POI_id": [0, 1, 2], "latitude": [0.0, 0.001, 0.002],
                              "longitude": [0.0, 0.0, 0.0]})
        mapping = position_mapping(frame, 200, 42)
        self.assertEqual(set(mapping), {0, 1, 2}); self.assertTrue(set(mapping.values()) <= {0, 1, 2})


if __name__ == "__main__": unittest.main()
