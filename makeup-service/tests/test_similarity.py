import json

import numpy as np
import pytest

from app.config import load_threshold
from app.identity import DetectedFace, cosine_similarity, largest_face
from app.ratelimit import RateLimiter


def test_identical_vectors_score_one():
    v = np.array([0.3, -1.2, 4.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_opposite_vectors_score_minus_one():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, -v) == pytest.approx(-1.0)


def test_orthogonal_vectors_score_zero():
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 5.0])) == pytest.approx(0.0)


def test_similarity_ignores_vector_length():
    a, b = np.array([1.0, 2.0, 2.0]), np.array([2.0, 1.0, 2.0])
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(10 * a, 0.1 * b))
    assert cosine_similarity(a, b) == pytest.approx(8 / 9)


def test_similarity_is_symmetric_on_face_sized_embeddings():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=512), rng.normal(size=512)
    assert cosine_similarity(a, b) == pytest.approx(cosine_similarity(b, a))
    assert -1.0 <= cosine_similarity(a, b) <= 1.0


def test_mismatched_shapes_are_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(512), np.ones(128))


def test_zero_embedding_is_rejected():
    with pytest.raises(ValueError):
        cosine_similarity(np.zeros(4), np.ones(4))


def test_largest_face_picks_biggest_box():
    small = DetectedFace((0, 0, 10, 10), 0.99, np.ones(2))
    big = DetectedFace((0, 0, 50, 40), 0.6, np.ones(2))
    assert largest_face([small, big]) is big
    assert largest_face([]) is None


def test_threshold_is_read_from_config(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 0.42, "method": "test"}))
    assert load_threshold(path) == pytest.approx(0.42)


def test_out_of_range_threshold_is_rejected(tmp_path):
    path = tmp_path / "threshold.json"
    path.write_text(json.dumps({"threshold": 3}))
    with pytest.raises(ValueError):
        load_threshold(path)


def test_rate_limiter_blocks_then_recovers():
    now = [0.0]
    limiter = RateLimiter(limit=2, window_seconds=10, clock=lambda: now[0])
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == 10
    assert limiter.check("b") is None
    now[0] = 10.5
    assert limiter.check("a") is None
