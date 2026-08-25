import unittest

import numpy as np

from hybrid.rq7_belief_memory import apply_variant, fuse, smoothed_counts, transition_prior


class RQ7BeliefMemoryTest(unittest.TestCase):
    def test_probabilities_are_normalized(self):
        actual = fuse(np.array([0.6, 0.4]), np.array([0.2, 0.8]), 0.5)
        self.assertAlmostEqual(float(actual.sum()), 1.0)
        self.assertTrue(np.all(actual >= 0))

    def test_history_and_transition_use_only_supplied_past(self):
        prior = np.array([0.5, 0.5])
        self.assertGreater(smoothed_counts([1, 1], prior, 1.0)[1], 0.5)
        self.assertGreater(transition_prior(0, {0: {1: 3}}, prior, 1.0)[1], 0.5)

    def test_sequential_state_resets_at_trajectory_boundary(self):
        base = np.array([[0.9, 0.1], [0.5, 0.5], [0.5, 0.5]])
        queries = [
            {"trajectory_id": "a", "prefix": [0]},
            {"trajectory_id": "a", "prefix": [0, 1]},
            {"trajectory_id": "b", "prefix": [1]},
        ]
        result = apply_variant(base, queries, "B2-sequential", 1.0, np.array([0.5, 0.5]), {}, 1.0)
        self.assertGreater(result[1, 0], 0.5)
        np.testing.assert_allclose(result[2], [0.5, 0.5])


if __name__ == "__main__": unittest.main()
