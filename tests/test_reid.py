"""
Unit Tests for ReIDFeatureExtractor and ReIDGallery (Phase 11).
"""
import numpy as np
import pytest
from animallens.reid.extractor import AnimalEmbedding, ReIDFeatureExtractor
from animallens.reid.gallery import IndividualProfile, ReIDGallery


def test_reid_feature_extractor():
    extractor = ReIDFeatureExtractor(embedding_dim=512)
    crop = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    emb = extractor.extract(crop, track_id=1)
    assert isinstance(emb, AnimalEmbedding)
    assert emb.track_id == 1
    assert emb.vector.shape == (512,)
    # Verify L2 normalization
    norm = np.linalg.norm(emb.vector)
    assert pytest.approx(norm, 0.01) == 1.0


def test_reid_similarity_computation():
    extractor = ReIDFeatureExtractor(embedding_dim=512)

    v1 = np.zeros(512, dtype=np.float32)
    v1[0] = 1.0

    v2 = np.zeros(512, dtype=np.float32)
    v2[0] = 1.0

    v3 = np.zeros(512, dtype=np.float32)
    v3[1] = 1.0

    # Identical vectors should have similarity 1.0
    assert pytest.approx(extractor.compute_similarity(v1, v2), 0.01) == 1.0
    # Orthogonal vectors should have similarity 0.0
    assert pytest.approx(extractor.compute_similarity(v1, v3), 0.01) == 0.0


def test_reid_gallery_registration_and_matching():
    gallery = ReIDGallery(match_threshold=0.85)

    # Register "Max"
    v_max = np.zeros(512, dtype=np.float32)
    v_max[0] = 1.0
    gallery.register("Max", v_max, species="dog", metadata={"breed": "Border Collie"})

    # Register "Bella"
    v_bella = np.zeros(512, dtype=np.float32)
    v_bella[10] = 1.0
    gallery.register("Bella", v_bella, species="dog", metadata={"breed": "Labrador"})

    assert len(gallery.profiles) == 2

    # Query with query vector close to Max
    query_max = np.zeros(512, dtype=np.float32)
    query_max[0] = 0.98
    query_max[1] = 0.10
    query_max /= np.linalg.norm(query_max)

    matched_name, sim = gallery.identify(query_max)
    assert matched_name == "Max"
    assert sim > 0.90

    # Query with unknown animal
    query_unknown = np.zeros(512, dtype=np.float32)
    query_unknown[100] = 1.0
    matched_unknown, sim_unknown = gallery.identify(query_unknown)
    assert matched_unknown is None
    assert sim_unknown < 0.85


def test_reid_match_or_create():
    gallery = ReIDGallery(match_threshold=0.85)
    crop = np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)

    name1, sim1 = gallery.match_or_create(crop, track_id=1, default_prefix="DOG")
    assert name1 == "DOG-01"
    assert len(gallery.profiles) == 1

    # Matching same crop should retrieve DOG-01
    name2, sim2 = gallery.match_or_create(crop, track_id=1, default_prefix="DOG")
    assert name2 == "DOG-01"
    assert sim2 > 0.85
