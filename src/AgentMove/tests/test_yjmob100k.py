import unittest

import pandas as pd

from hybrid.prepare_dataset import assign_trajectories
from processing.prepare_yjmob100k import canonical_columns, stable_sample


class YJMob100KTests(unittest.TestCase):
    def test_yjmob_aliases_are_normalized(self) -> None:
        frame = canonical_columns(pd.DataFrame({"uid": [1], "d": [2], "t": [3], "x": [4], "y": [5]}))
        self.assertTrue({"user_id", "day", "timeslot", "x", "y"}.issubset(frame.columns))

    def test_user_sampling_is_deterministic(self) -> None:
        users = [str(value) for value in range(100)]
        self.assertEqual(stable_sample(users, 10, 42), stable_sample(users, 10, 42))
        self.assertNotEqual(stable_sample(users, 10, 42), stable_sample(users, 10, 43))

    def test_daily_trajectory_ids_are_preserved(self) -> None:
        frame = pd.DataFrame({"user_id": ["u", "u"], "traj_id": ["4", "5"]})
        result = assign_trajectories(frame, "yjmob100k", 72)
        self.assertEqual(result["trajectory_id"].tolist(), ["u_4", "u_5"])


if __name__ == "__main__":
    unittest.main()
