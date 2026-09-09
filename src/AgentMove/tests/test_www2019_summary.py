from pathlib import Path

import numpy as np

from hybrid.www2019_summary import PAIRED_COMPARISONS, paired_summary


def _write_prediction(path: Path, ranks: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(ranks)
    np.savez_compressed(
        path,
        query_index=np.arange(len(values)),
        labels=np.arange(len(values)),
        ranks=values,
        reciprocal_rank=1.0 / values,
        true_probability=np.where(values == 1, 0.8, 0.2),
        brier=np.where(values == 1, 0.1, 0.9),
    )


def test_paired_summary_covers_locked_www2019_comparisons(tmp_path: Path) -> None:
    paths = {
        (variant, order)
        for _, left_variant, left_order, right_variant, right_order in PAIRED_COMPARISONS
        for variant, order in ((left_variant, left_order), (right_variant, right_order))
    }
    rank_sets = {
        ("E0-ce", "correct"): [2, 2, 2, 2],
        ("E1-kd", "correct"): [1, 2, 1, 2],
        ("E5-dual", "correct"): [1, 1, 1, 2],
        ("E5-dual", "reverse"): [2, 2, 1, 2],
        ("E5-dual", "random"): [3, 2, 2, 2],
    }
    for variant, order in paths:
        _write_prediction(
            tmp_path / variant / order / "seed-42" / "test.predictions.npz",
            rank_sets[(variant, order)],
        )

    rows = paired_summary(tmp_path, [42], iterations=1000)

    assert len(rows) == len(PAIRED_COMPARISONS) * 6
    recall = {
        row["comparison"]: row["effect_favoring_first"]
        for row in rows if row["metric"] == "recall@1"
    }
    assert recall["E1-kd-vs-E0-ce"] > 0
    assert recall["E5-dual-vs-E1-kd"] > 0
    assert recall["correct-vs-reverse"] > 0
    assert recall["correct-vs-random"] > 0
    assert all(0.0 <= row["holm_adjusted_p"] <= 1.0 for row in rows)
