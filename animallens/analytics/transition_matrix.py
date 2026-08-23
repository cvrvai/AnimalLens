"""
Markov behavioral state transition probability matrix calculator.
Operations Research & Stochastic Process Modeling for Animal Behavior.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from animallens.core.schemas import BehaviorEvent


class TransitionMatrixResult(BaseModel):
    """Result container for behavior state transition dynamics."""
    states: List[str] = Field(..., description="Ordered list of unique behavioral states")
    matrix: List[List[float]] = Field(..., description="Row-stochastic transition probability matrix P[i][j]")
    transition_counts: Dict[str, Dict[str, int]] = Field(..., description="Raw observed transition counts")
    total_transitions: int = Field(..., description="Total count of observed state transitions")
    stationary_distribution: Optional[Dict[str, float]] = Field(
        default=None, description="Steady-state asymptotic distribution pi"
    )

    def to_dict_matrix(self) -> Dict[str, Dict[str, float]]:
        """Dictionary representation mapping state_from -> {state_to: probability}."""
        res = {}
        for i, s_from in enumerate(self.states):
            res[s_from] = {}
            for j, s_to in enumerate(self.states):
                res[s_from][s_to] = round(self.matrix[i][j], 4)
        return res


def compute_transition_matrix(events: List[BehaviorEvent]) -> TransitionMatrixResult:
    """
    Compute first-order Markov transition probability matrix P[i][j] from temporal behavior events.
    P[i][j] = P(State_{t+1} = j | State_t = i)
    """
    if not events or len(events) < 2:
        return TransitionMatrixResult(
            states=[],
            matrix=[],
            transition_counts={},
            total_transitions=0,
            stationary_distribution={},
        )

    # Extract sequential labels
    sequence = [f"{e.behavior.category}.{e.behavior.label}" for e in events]
    unique_states = sorted(list(set(sequence)))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n_states = len(unique_states)

    count_matrix = np.zeros((n_states, n_states), dtype=int)
    counts_dict: Dict[str, Dict[str, int]] = {s: {s2: 0 for s2 in unique_states} for s in unique_states}

    for t in range(len(sequence) - 1):
        s_curr = sequence[t]
        s_next = sequence[t + 1]
        i = state_to_idx[s_curr]
        j = state_to_idx[s_next]
        count_matrix[i, j] += 1
        counts_dict[s_curr][s_next] += 1

    total_transitions = int(np.sum(count_matrix))

    # Normalize rows to create stochastic matrix
    prob_matrix = np.zeros((n_states, n_states), dtype=float)
    for i in range(n_states):
        row_sum = np.sum(count_matrix[i, :])
        if row_sum > 0:
            prob_matrix[i, :] = count_matrix[i, :] / row_sum
        else:
            # Self-loop default if state is absorbing
            prob_matrix[i, i] = 1.0

    # Compute stationary distribution if transition graph is ergodic
    stationary_dist: Dict[str, float] = {}
    try:
        # Stationary vector pi satisfies pi * P = pi and sum(pi) = 1
        eigenvalues, eigenvectors = np.linalg.eig(prob_matrix.T)
        close_to_1 = np.isclose(eigenvalues, 1.0)
        if np.any(close_to_1):
            idx = np.where(close_to_1)[0][0]
            pi = np.real(eigenvectors[:, idx])
            pi = pi / np.sum(pi)
            for k, state_name in enumerate(unique_states):
                stationary_dist[state_name] = round(float(pi[k]), 4)
    except Exception:
        pass

    return TransitionMatrixResult(
        states=unique_states,
        matrix=prob_matrix.tolist(),
        transition_counts=counts_dict,
        total_transitions=total_transitions,
        stationary_distribution=stationary_dist if stationary_dist else None,
    )
