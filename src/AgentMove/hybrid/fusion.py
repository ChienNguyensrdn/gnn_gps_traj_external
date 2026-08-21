from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .bayesian_network import TrajectoryBayesianNetwork


@dataclass
class BayesianEvidenceFusion:
    epsilon: float = 1e-9

    def fuse(
        self,
        candidate_ids: Iterable[str],
        priors: Iterable[float],
        habit_likelihoods: Iterable[float],
        semantic_likelihoods: Iterable[float],
    ) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        network = TrajectoryBayesianNetwork(candidate_ids, epsilon=self.epsilon)
        network.set_prior(priors)
        network.observe("H", habit_likelihoods)
        network.observe("S", semantic_likelihoods)
        return network.infer()
