"""Shared helpers for the evaluation scripts. Run the scripts from the makeup-service folder."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SERVICE_ROOT = Path(__file__).resolve().parent.parent
if str(SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVICE_ROOT))

from app.config import MAX_UPLOAD_BYTES, Settings  # noqa: E402
from app.identity import FaceEmbedder, InsightFaceEmbedder  # noqa: E402
from app.uploads import UploadError, load_image, to_bgr  # noqa: E402

EVAL_DIR = SERVICE_ROOT / "eval"
FACES_DIR = EVAL_DIR / "faces"
RESULTS_DIR = EVAL_DIR / "results"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class Face:
    person: str
    path: Path
    image: object = None  # PIL image, as the API would see it
    embedding: np.ndarray | None = None
    problem: str | None = None

    @property
    def name(self) -> str:
        return f"{self.person}/{self.path.name}"


def list_faces(faces_dir: Path = FACES_DIR) -> list[Face]:
    """eval/faces/<person>/<photo>: one folder per person, any number of photos inside."""
    faces = [
        Face(person.name, path)
        for person in sorted(p for p in faces_dir.iterdir() if p.is_dir())
        for path in sorted(person.iterdir())
        if path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not faces:
        sys.exit(f"No photos found. Put them in {faces_dir}/<person>/<photo>.jpg")
    return faces


def embed_faces(faces: list[Face], embedder: FaceEmbedder) -> list[Face]:
    """Loads each photo exactly as the API does and keeps it only if it has exactly one face."""
    for face in faces:
        try:
            face.image = load_image(face.path.read_bytes(), MAX_UPLOAD_BYTES)
        except UploadError as error:
            face.problem = error.code
            continue
        found = embedder.faces(to_bgr(face.image))
        if len(found) != 1:
            face.problem = "no_face" if not found else "multiple_faces"
            continue
        face.embedding = found[0].embedding
    return faces


def make_embedder() -> InsightFaceEmbedder:
    return InsightFaceEmbedder(os.environ.get("INSIGHTFACE_ROOT", Settings.model_root))
