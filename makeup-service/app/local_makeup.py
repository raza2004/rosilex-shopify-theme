"""Local baseline engine (GENERATOR=local): paints a preset style along MediaPipe face mesh landmarks.

This is not generative AI and is not used by the store. It exists for evaluation: it costs nothing,
and because it only recolours lips, cheeks and eyelids it shows how identity scores behave for an
edit that cannot change face shape, as a baseline next to the generative engine. Product looks are
not supported, only the five preset styles.
"""

from __future__ import annotations

import io
import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .generator import EditRequest, GeneratedImage, GenerationError
from .regions import CHEEKS, EYE_OPENINGS, FACE_EDGES, LIPS_INNER, LIPS_OUTER

# Upper lid from outer to inner corner, and the lower edge of the brow above it, outer to inner.
EYE_LIDS = (
    ([33, 246, 161, 160, 159, 158, 157, 173, 133], [46, 53, 52, 65, 55]),
    ([263, 466, 388, 387, 386, 385, 384, 398, 362], [276, 283, 282, 295, 285]),
)


@dataclass(frozen=True)
class Layer:
    area: str  # lips, blush, shadow, liner
    color: tuple[int, int, int]  # RGB
    strength: float
    size: float = 1.0  # shadow height (share of lid to brow distance)


LOOKS: dict[str, tuple[Layer, ...]] = {
    "natural": (Layer("lips", (190, 110, 110), 0.28), Layer("blush", (225, 130, 130), 0.14), Layer("liner", (70, 45, 35), 0.35)),
    "soft_glam": (
        Layer("shadow", (205, 165, 130), 0.30, 0.55), Layer("liner", (40, 25, 20), 0.6),
        Layer("blush", (235, 140, 115), 0.22), Layer("lips", (200, 115, 115), 0.42),
    ),
    "bold_lip": (Layer("lips", (175, 20, 35), 0.72), Layer("liner", (35, 25, 25), 0.35)),
    "smoky_eye": (Layer("shadow", (55, 50, 58), 0.48, 0.75), Layer("liner", (15, 12, 15), 0.8), Layer("lips", (175, 125, 115), 0.25)),
    "bridal": (
        Layer("shadow", (215, 150, 140), 0.28, 0.5), Layer("liner", (60, 38, 30), 0.5),
        Layer("blush", (240, 150, 160), 0.2), Layer("lips", (210, 120, 135), 0.48),
    ),
}


def _resample(points: np.ndarray, count: int) -> np.ndarray:
    segments = np.linalg.norm(np.diff(points, axis=0), axis=1)
    distance = np.concatenate([[0], np.cumsum(segments)])
    targets = np.linspace(0, distance[-1], count)
    return np.stack([np.interp(targets, distance, points[:, i]) for i in range(2)], axis=1)


def _feather(mask: np.ndarray, radius: float) -> np.ndarray:
    k = max(3, int(radius) * 2 + 1)
    return cv2.GaussianBlur(mask, (k, k), 0)


def _tint(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], strength: float) -> np.ndarray:
    """Blends colour in, scaled by each pixel's brightness so texture stays visible."""
    weight = (mask * strength)[..., None]
    if weight.max() <= 0:
        return image
    luminance = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    reference = float((luminance * mask).sum() / max(mask.sum(), 1e-6))
    shade = np.clip(luminance / max(reference, 1e-3), 0.4, 1.6)[..., None]
    target = np.clip(np.array(color, dtype=np.float32) / 255.0 * shade, 0, 1)
    return image * (1 - weight) + target * weight


class LocalMakeupGenerator:
    engine = "local"

    def __init__(self):
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self._lock = threading.Lock()

    def generate(self, request: EditRequest) -> GeneratedImage:
        layers = LOOKS.get(request.look_id)
        if layers is None:
            raise GenerationError(f"the local engine has no look '{request.look_id}'", retryable=False)
        rgb = np.asarray(Image.open(io.BytesIO(request.image_png)).convert("RGB"))
        height, width = rgb.shape[:2]
        with self._lock:
            found = self._mesh.process(rgb)
        if not found.multi_face_landmarks:
            raise GenerationError("no face landmarks found", retryable=False)
        points = np.array([(p.x * width, p.y * height) for p in found.multi_face_landmarks[0].landmark], dtype=np.float32)
        face_width = float(np.linalg.norm(points[FACE_EDGES[0]] - points[FACE_EDGES[1]]))

        image = rgb.astype(np.float32) / 255.0
        for layer in layers:
            image = _tint(image, self._mask(layer, points, (height, width), face_width), layer.color, layer.strength)
        buffer = io.BytesIO()
        Image.fromarray((np.clip(image, 0, 1) * 255).round().astype(np.uint8)).save(buffer, format="PNG")
        return GeneratedImage(buffer.getvalue(), "image/png")

    @staticmethod
    def _mask(layer: Layer, points: np.ndarray, shape: tuple[int, int], face_width: float) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.float32)
        if layer.area == "lips":
            cv2.fillPoly(mask, [points[LIPS_OUTER].round().astype(np.int32)], 1.0)
            cv2.fillPoly(mask, [points[LIPS_INNER].round().astype(np.int32)], 0.0)
            return _feather(mask, face_width * 0.012)
        if layer.area == "blush":
            radius = int(face_width * 0.13)
            for index in CHEEKS:
                cv2.circle(mask, tuple(points[index].round().astype(int)), radius, 1.0, -1)
            return _feather(mask, radius * 0.9)
        for lid_ids, brow_ids in EYE_LIDS:
            lid = points[lid_ids]
            if layer.area == "shadow":
                brow = _resample(points[brow_ids], len(lid_ids))
                polygon = np.concatenate([lid, (lid + (brow - lid) * layer.size)[::-1]]).round().astype(np.int32)
                cv2.fillPoly(mask, [polygon], 1.0)
            elif layer.area == "liner":
                cv2.polylines(mask, [lid.round().astype(np.int32)], False, 1.0, max(1, int(face_width * 0.007)), cv2.LINE_AA)
        for ids in EYE_OPENINGS:  # never paint the eyeballs
            cv2.fillPoly(mask, [points[ids].round().astype(np.int32)], 0.0)
        return _feather(mask, face_width * (0.04 if layer.area == "shadow" else 0.004))
