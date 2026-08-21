"""Explicit Bayesian network used by Stage 3.

The calibrated CGM distribution Q is the prior P(L), not a second child node.
The probabilistic graph is therefore L -> H and L -> S. This avoids counting
the CGM signal twice while retaining the paper's conditional-independence model
H independent of S given L.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np


@dataclass(frozen=True)
class BayesianNode:
    name: str
    kind: str
    states: Tuple[str, ...]
    observed: bool
    description: str


@dataclass(frozen=True)
class BayesianEdge:
    parent: str
    child: str


class TrajectoryBayesianNetwork:
    """Naive-Bayes network with exact discrete inference over locations."""

    def __init__(self, candidate_ids: Iterable[str], epsilon: float = 1e-9) -> None:
        ids = tuple(str(value) for value in candidate_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("BBN candidate states must be non-empty and unique")
        self.epsilon = epsilon
        self.nodes = {
            "L": BayesianNode("L", "latent", ids, False, "True next-location candidate"),
            "H": BayesianNode("H", "evidence", ("fit",), True, "Personal-habit evidence"),
            "S": BayesianNode("S", "evidence", ("fit",), True, "Urban-semantic evidence"),
        }
        self.edges = (BayesianEdge("L", "H"), BayesianEdge("L", "S"))
        self.prior_source = "Q: calibrated CGM distribution"
        self.prior: np.ndarray | None = None
        self.likelihoods: Dict[str, np.ndarray] = {}

    @property
    def candidate_ids(self) -> Tuple[str, ...]:
        return self.nodes["L"].states

    def set_prior(self, probabilities: Iterable[float]) -> "TrajectoryBayesianNetwork":
        values = np.asarray(list(probabilities), dtype=float)
        self._validate_vector(values, "prior")
        if np.any(values < 0) or values.sum() <= 0:
            raise ValueError("BBN prior must be non-negative with positive mass")
        self.prior = values / values.sum()
        return self

    def observe(self, node: str, likelihood_by_location: Iterable[float]) -> "TrajectoryBayesianNetwork":
        if node not in {"H", "S"}:
            raise ValueError(f"Only H and S are observable likelihood nodes, got {node}")
        values = np.asarray(list(likelihood_by_location), dtype=float)
        self._validate_vector(values, f"likelihood {node}")
        if np.any(values < 0):
            raise ValueError(f"Likelihood {node} must be non-negative")
        # Likelihood ratios/densities may exceed one; only positivity matters
        # because inference normalizes the joint scores across candidates.
        self.likelihoods[node] = np.clip(values, self.epsilon, None)
        return self

    def infer(self) -> Tuple[np.ndarray, Dict[str, Dict[str, float]]]:
        if self.prior is None:
            raise RuntimeError("BBN prior P(L) has not been set")
        missing = {"H", "S"} - set(self.likelihoods)
        if missing:
            raise RuntimeError(f"BBN missing observed evidence nodes: {sorted(missing)}")
        log_prior = np.log(np.clip(self.prior, self.epsilon, 1.0))
        log_habit = np.log(self.likelihoods["H"])
        log_semantic = np.log(self.likelihoods["S"])
        joint_log = log_prior + log_habit + log_semantic
        maximum = float(joint_log.max())
        shifted = joint_log - maximum
        log_normalizer = maximum + float(np.log(np.exp(shifted).sum()))
        posterior = np.exp(joint_log - log_normalizer)
        contributions = {
            candidate_id: {
                "log_prior": float(log_prior[index]),
                "log_habit": float(log_habit[index]),
                "log_semantic": float(log_semantic[index]),
                "log_joint_unnormalized": float(joint_log[index]),
                "log_normalizer": log_normalizer,
                "log_posterior": float(joint_log[index] - log_normalizer),
            }
            for index, candidate_id in enumerate(self.candidate_ids)
        }
        return posterior, contributions

    def structure(self) -> Dict[str, object]:
        return {
            "model": "conditional-independent naive Bayesian network",
            "prior_source": self.prior_source,
            "nodes": [
                {
                    "name": node.name,
                    "kind": node.kind,
                    "states": list(node.states),
                    "observed": node.observed,
                    "description": node.description,
                }
                for node in self.nodes.values()
            ],
            "edges": [{"parent": edge.parent, "child": edge.child} for edge in self.edges],
            "factorization": "P(L|H,S) proportional to P(L;Q) * P(H|L) * P(S|L)",
            "independence_assumption": "H independent of S given L",
        }

    def _validate_vector(self, values: np.ndarray, label: str) -> None:
        if values.shape != (len(self.candidate_ids),):
            raise ValueError(f"BBN {label} must have {len(self.candidate_ids)} values, got {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError(f"BBN {label} contains non-finite values")
