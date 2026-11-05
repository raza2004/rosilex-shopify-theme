"""Fakes for Gemini and InsightFace, so tests run offline and deterministically.

Test images are solid colours. The red channel says how many faces the fake detector finds
(low: none, middle: one, high: two) and the green channel sets the angle of the face embedding,
so an image can be built to have an exact cosine similarity to the original.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.generator import EditRequest, GeneratedImage, GenerationError
from app.identity import DetectedFace
from app.main import create_app
from app.regions import NoLandmarksError, RegionMasks

NO_FACE, ONE_FACE, TWO_FACES = 20, 120, 230
THRESHOLD = 0.5


def green_for(similarity: float) -> int:
    """Green value whose embedding has this cosine similarity to an image with green 0."""
    return round(math.acos(similarity) / (math.pi / 2) * 255)


def solid(red: int, green: int = 0, size: int = 64, fmt: str = "PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (size, size), (red, green, 0)).save(buffer, format=fmt)
    return buffer.getvalue()


def face_image(similarity: float | None) -> bytes:
    """A generated result: None means no face in it."""
    if similarity is None:
        return solid(NO_FACE)
    return solid(ONE_FACE, green_for(similarity))


class FakeEmbedder:
    def faces(self, bgr: np.ndarray) -> list[DetectedFace]:
        red, green = float(bgr[..., 2].mean()), float(bgr[..., 1].mean())
        theta = green / 255 * (math.pi / 2)
        embedding = np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)
        count = 0 if red < 70 else 1 if red < 180 else 2
        return [DetectedFace((0, 0, 10 + i, 10 + i), 0.9, embedding) for i in range(count)]


class FakeGenerator:
    """Returns the queued results in order; an Exception in the queue is raised as a Gemini failure."""

    engine = "fake"

    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0
        self.prompts: list[str] = []
        self.requests: list[EditRequest] = []

    def generate(self, request: EditRequest) -> GeneratedImage:
        self.calls += 1
        self.prompts.append(request.prompt)
        self.requests.append(request)
        item = self.results.pop(0)
        if isinstance(item, GenerationError):
            raise item
        if isinstance(item, Exception):
            raise GenerationError(str(item))
        return GeneratedImage(face_image(item), "image/png")


class FakeRegions:
    """Marks the whole photo as a makeup region with nothing protected, so the composite is the generated image."""

    def __init__(self, landmarks: bool = True):
        self.landmarks = landmarks
        self.requested: list[set[str]] = []

    def masks(self, image, regions: set[str]) -> RegionMasks:
        if not self.landmarks:
            raise NoLandmarksError()
        self.requested.append(set(regions))
        shape = (image.height, image.width)
        return RegionMasks(np.ones(shape, np.float32), np.zeros(shape, np.float32), np.zeros(shape, np.float32))


def make_settings(**overrides) -> Settings:
    values = dict(
        gemini_api_key="test",
        allowed_origins=["https://rosexious.myshopify.com"],
        threshold=THRESHOLD,
        rate_limit_requests=100,
        rate_limit_window_seconds=60,
        daily_generation_limit=0,  # off unless a test turns it on (see test_quota.py)
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def client_for():
    def build(generator=None, regions=None, **overrides) -> TestClient:
        settings = make_settings(**overrides)
        return TestClient(create_app(settings, generator=generator, embedder=FakeEmbedder(), regions=regions or FakeRegions()))

    return build


@pytest.fixture
def upload():
    """Form fields for POST /api/makeup with a one-face photo by default."""

    def build(data: bytes | None = None, style: str | None = "natural", product: dict | str | None = None,
              filename: str = "me.jpg", content_type: str = "image/jpeg"):
        data = solid(ONE_FACE, 0, fmt="JPEG") if data is None else data
        fields = {}
        if style is not None:
            fields["style"] = style
        if product is not None:
            fields["product"] = product if isinstance(product, str) else json.dumps(product)
        return {"files": {"image": (filename, data, content_type)}, "data": fields}

    return build
