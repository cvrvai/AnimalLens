"""
Spatial metrics and multi-agent kinematic dispersion analysis.
Operations Research & Spatial Graph Analytics for animal behavior.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from animallens.core.schemas import BehaviorEvent, SubjectInfo


class SpatialMetricsResult(BaseModel):
    """Container for multi-animal spatial dispersion and proximity metrics."""
    mean_inter_individual_distance: float = Field(..., description="Average pairwise distance between subjects")
    min_inter_individual_distance: float = Field(..., description="Closest approach distance observed")
    nearest_neighbor_distances: List[float] = Field(default_factory=list, description="Distance to closest conspecific for each subject")
    clark_evans_dispersion_index: float = Field(..., description="R index: <1 aggregated/clustered, ~1 random, >1 dispersed/territorial")
    crowding_intensity: float = Field(..., description="Spatial density index within active arena [0.0, 1.0]")
    active_subjects_count: int = Field(..., description="Number of tracked subjects analyzed")


def compute_spatial_metrics(subjects: List[SubjectInfo], arena_area: float = 1.0) -> SpatialMetricsResult:
    """
    Compute spatial graph analytics, Nearest Neighbor Distance (NND), and Clark-Evans dispersion index.
    """
    valid_subjects = [s for s in subjects if s.bbox is not None]
    n = len(valid_subjects)

    if n < 2:
        return SpatialMetricsResult(
            mean_inter_individual_distance=0.0,
            min_inter_individual_distance=0.0,
            nearest_neighbor_distances=[],
            clark_evans_dispersion_index=1.0,
            crowding_intensity=0.0,
            active_subjects_count=n,
        )

    # Extract centroids
    centroids = np.array([s.bbox.center for s in valid_subjects])  # Shape: (N, 2)

    # Pairwise Euclidean distance matrix
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # Mask diagonal (self-distance)
    np.fill_diagonal(dist_matrix, np.inf)

    # Nearest neighbor distance for each individual
    nnd = np.min(dist_matrix, axis=1)
    mean_nnd = float(np.mean(nnd))

    # All pairwise off-diagonal distances
    upper_tri_indices = np.triu_indices(n, k=1)
    pairwise_distances = dist_matrix[upper_tri_indices]
    mean_iid = float(np.mean(pairwise_distances))
    min_iid = float(np.min(pairwise_distances))

    # Clark-Evans Dispersion Index (R)
    # Expected mean NND under complete spatial randomness (CSR) is 1 / (2 * sqrt(density))
    density = n / max(0.01, arena_area)
    expected_nnd_csr = 1.0 / (2.0 * math.sqrt(density))
    r_index = mean_nnd / max(0.001, expected_nnd_csr)

    # Crowding intensity based on proximity threshold (< 0.20 normalized units)
    close_pairs = np.sum(pairwise_distances < 0.20)
    total_pairs = len(pairwise_distances)
    crowding = float(close_pairs / max(1, total_pairs))

    return SpatialMetricsResult(
        mean_inter_individual_distance=round(mean_iid, 4),
        min_inter_individual_distance=round(min_iid, 4),
        nearest_neighbor_distances=[round(float(d), 4) for d in nnd],
        clark_evans_dispersion_index=round(r_index, 4),
        crowding_intensity=round(crowding, 4),
        active_subjects_count=n,
    )
