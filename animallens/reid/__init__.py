"""
Deep Metric Re-Identification (ReID) package for AnimalLens.
"""
from animallens.reid.extractor import AnimalEmbedding, ReIDFeatureExtractor
from animallens.reid.gallery import IndividualProfile, ReIDGallery

__all__ = [
    "AnimalEmbedding",
    "ReIDFeatureExtractor",
    "IndividualProfile",
    "ReIDGallery",
]
