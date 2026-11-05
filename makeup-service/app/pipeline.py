"""Generate, composite, measure identity, retry: the core loop shared by the API and the evaluation."""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from PIL import Image, UnidentifiedImageError

from . import compositing
from .generator import EditRequest, GenerationError, ImageGenerator
from .identity import FaceEmbedder, cosine_similarity, largest_face
from .regions import RegionMasks
from .styles import Look
from .uploads import to_bgr


@dataclass
class Attempt:
    number: int
    image: bytes | None = None  # composited result, PNG
    raw: bytes | None = None  # the image model's own output, before compositing (debugging only)
    mime_type: str | None = None
    score: float | None = None
    error: str | None = None

    @property
    def face_detected(self) -> bool:
        return self.score is not None


@dataclass
class PipelineResult:
    attempts: list[Attempt] = field(default_factory=list)
    threshold: float = 0.0
    quota_exceeded: bool = False

    @property
    def best(self) -> Attempt | None:
        """The highest scoring attempt; an image with no detected face only if nothing scored."""
        scored = [a for a in self.attempts if a.score is not None]
        if scored:
            return max(scored, key=lambda a: a.score)
        with_image = [a for a in self.attempts if a.image is not None]
        return with_image[-1] if with_image else None

    @property
    def passed(self) -> bool:
        best = self.best
        return bool(best and best.score is not None and best.score >= self.threshold)

    @property
    def detection_failures(self) -> int:
        return sum(1 for a in self.attempts if a.image is not None and a.score is None)

    @property
    def generation_failures(self) -> int:
        return sum(1 for a in self.attempts if a.image is None)


def score_image(original_embedding: np.ndarray, image: Image.Image, embedder: FaceEmbedder) -> float | None:
    """Similarity between the original face and the largest face in the image, or None if no face."""
    face = largest_face(embedder.faces(to_bgr(image)))
    return None if face is None else cosine_similarity(original_embedding, face.embedding)


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def run(
    image: Image.Image,
    original_embedding: np.ndarray,
    look: Look,
    masks: RegionMasks,
    generator: ImageGenerator,
    embedder: FaceEmbedder,
    threshold: float,
    max_attempts: int,
) -> PipelineResult:
    """Tries up to max_attempts times, stopping at the first attempt that passes the threshold.

    masks (see app.regions) limit what the model may edit, and decide what of its output is
    composited onto the original: makeup pixels, the skin's tone change, nothing from protected features.
    """
    result = PipelineResult(threshold=threshold)
    prepared = compositing.prepare(image, masks)
    request = EditRequest(prepared.image_png, prepared.mask_png, prepared.size, look.prompt, look.id)

    for number in range(1, max_attempts + 1):
        attempt = Attempt(number)
        result.attempts.append(attempt)
        try:
            generated = generator.generate(request)
        except GenerationError as error:
            attempt.error = str(error)
            if error.quota:
                result.quota_exceeded = True
            if not error.retryable:
                break
            continue
        try:
            final = compositing.composite(image, generated.data, prepared)
        except (UnidentifiedImageError, OSError):
            attempt.error = "the engine returned an unreadable image"
            continue
        attempt.image, attempt.mime_type, attempt.raw = _png(final), "image/png", generated.data
        attempt.score = score_image(original_embedding, final, embedder)
        if attempt.score is None:
            attempt.error = "no face detected in the result"
        elif attempt.score >= threshold:
            break

    return result
