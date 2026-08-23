"""
ReID Subject Gallery & Identity Management Database for AnimalLens.
Maintains persistent individual profiles ('Max', 'Bella', 'ALPHA-01') across video cuts and days.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from animallens.reid.extractor import AnimalEmbedding, ReIDFeatureExtractor

logger = logging.getLogger(__name__)


@dataclass
class IndividualProfile:
    """Persistent Animal Identity Profile."""
    name: str
    species: str
    anchor_embedding: np.ndarray
    sample_count: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_embedding(self, new_emb: np.ndarray, alpha: float = 0.1) -> None:
        """Exponential Moving Average (EMA) update for adaptive appearance tracking."""
        updated = (1.0 - alpha) * self.anchor_embedding + alpha * new_emb
        norm = np.linalg.norm(updated) + 1e-6
        self.anchor_embedding = (updated / norm).astype(np.float32)
        self.sample_count += 1


class ReIDGallery:
    """
    Gallery vector store for individual animal re-identification.
    """

    def __init__(self, match_threshold: float = 0.80) -> None:
        self.match_threshold = match_threshold
        self.profiles: Dict[str, IndividualProfile] = {}
        self.extractor = ReIDFeatureExtractor()

    def register(
        self,
        name: str,
        embedding: Union[np.ndarray, AnimalEmbedding],
        species: str = "dog",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IndividualProfile:
        """Register a known named animal in the gallery."""
        vec = embedding.vector if isinstance(embedding, AnimalEmbedding) else embedding
        norm = np.linalg.norm(vec) + 1e-6
        normalized_vec = (vec / norm).astype(np.float32)

        profile = IndividualProfile(
            name=name,
            species=species,
            anchor_embedding=normalized_vec,
            metadata=metadata or {},
        )
        self.profiles[name] = profile
        logger.info(f"Registered animal identity: {name} ({species}) into ReID gallery.")
        return profile

    def identify(
        self,
        embedding: Union[np.ndarray, AnimalEmbedding],
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Identifies an animal by comparing its embedding against all gallery profiles.
        Returns (best_matched_name, highest_similarity_score).
        """
        vec = embedding.vector if isinstance(embedding, AnimalEmbedding) else embedding
        thresh = threshold or self.match_threshold

        best_name: Optional[str] = None
        best_sim: float = -1.0

        for name, prof in self.profiles.items():
            sim = self.extractor.compute_similarity(vec, prof.anchor_embedding)
            if sim > best_sim:
                best_sim = sim
                if sim >= thresh:
                    best_name = name

        return (best_name, best_sim)

    def match_or_create(
        self,
        crop: Any,
        track_id: int,
        default_prefix: str = "DOG",
    ) -> Tuple[str, float]:
        """
        Extracts embedding from crop and either matches an existing individual or assigns a new persistent ID.
        """
        emb = self.extractor.extract(crop, track_id=track_id)
        matched_name, sim = self.identify(emb)

        if matched_name:
            # Update EMA embedding
            self.profiles[matched_name].update_embedding(emb.vector)
            return (matched_name, sim)

        # Auto-create new individual profile
        new_name = f"{default_prefix}-{track_id:02d}"
        self.register(new_name, emb)
        return (new_name, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_registered": len(self.profiles),
            "profiles": [
                {
                    "name": p.name,
                    "species": p.species,
                    "sample_count": p.sample_count,
                    "metadata": p.metadata,
                }
                for p in self.profiles.values()
            ],
        }
