"""Face detection and identity embeddings with InsightFace buffalo_l on CPU."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    bbox: tuple[float, float, float, float]
    det_score: float
    embedding: np.ndarray

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


class FaceEmbedder(Protocol):
    def faces(self, bgr: np.ndarray) -> list[DetectedFace]: ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two embeddings, in [-1, 1]."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"embedding shapes differ: {a.shape} vs {b.shape}")
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        raise ValueError("cannot compare a zero embedding")
    return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))


def largest_face(faces: list[DetectedFace]) -> DetectedFace | None:
    return max(faces, key=lambda face: face.area, default=None)


class InsightFaceEmbedder:
    """Wraps insightface.app.FaceAnalysis, loading only the detection and recognition models."""

    def __init__(self, model_root: str, det_size: int = 640):
        from insightface.app import FaceAnalysis

        self._app = FaceAnalysis(
            name="buffalo_l",
            root=model_root,
            providers=["CPUExecutionProvider"],
            allowed_modules=["detection", "recognition"],
        )
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def faces(self, bgr: np.ndarray) -> list[DetectedFace]:
        return [
            DetectedFace(
                bbox=tuple(float(v) for v in face.bbox),
                det_score=float(face.det_score),
                embedding=np.asarray(face.normed_embedding, dtype=np.float32),
            )
            for face in self._app.get(bgr)
        ]
