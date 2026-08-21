import unittest

import numpy as np

from hybrid.tist2015_markov import stable_descending_rank


class MarkovRankTest(unittest.TestCase):
    def test_rank_matches_stable_descending_ties(self):
        scores = np.asarray([0.5, 0.9, 0.5, 0.1])
        self.assertEqual(stable_descending_rank(scores, 0), 2)
        self.assertEqual(stable_descending_rank(scores, 2), 3)
        self.assertEqual(stable_descending_rank(scores, 1), 1)


if __name__ == "__main__":
    unittest.main()
