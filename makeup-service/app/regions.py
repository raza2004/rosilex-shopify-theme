"""Makeup regions (lips, eyes, cheeks, skin) as image masks, from MediaPipe face mesh landmarks.

The masks are used twice: sent to the image model as the area it may edit, and used to
composite the result back onto the original photo, so pixels outside them stay original.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

REGIONS = ("lips", "eyes", "cheeks", "skin")

# MediaPipe face mesh indices (the subject's right side is on the image left).
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
EYE_OPENINGS = (
    [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
    [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466],
)
# Eye socket: lower edge of the brow (outer to inner), then the lower lid (inner to outer).
EYE_SOCKETS = (
    [46, 53, 52, 65, 55, 133, 155, 154, 153, 145, 144, 163, 7, 33],
    [276, 283, 282, 295, 285, 362, 382, 381, 380, 374, 373, 390, 249, 263],
)
BROWS = (
    [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]
CHEEKS = (50, 280)
FACE_EDGES = (234, 454)


class NoLandmarksError(Exception):
    """MediaPipe found no face mesh in the photo."""


class FaceRegions:
    def __init__(self):
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        self._lock = threading.Lock()  # FaceMesh is not safe to call from several threads at once

    def landmarks(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"))
        with self._lock:
            found = self._mesh.process(rgb)
        if not found.multi_face_landmarks:
            raise NoLandmarksError()
        height, width = rgb.shape[:2]
        return np.array([(p.x * width, p.y * height) for p in found.multi_face_landmarks[0].landmark], dtype=np.float32)

    def masks(self, image: Image.Image, regions: set[str]) -> "RegionMasks":
        """The makeup, skin and protected masks for the requested regions."""
        return region_masks(self.landmarks(image), (image.height, image.width), regions)


@dataclass(frozen=True)
class RegionMasks:
    """Masks at the photo's resolution (float32, height x width).

    makeup:  lips, eyes or cheeks to recolour, binary. Generated pixels are pasted here.
    skin:    facial skin, binary. Only the soft tone change (glow, evenness) is taken here, never
             edges or detail, so features the model moved cannot appear twice.
    protect: soft 0..1. Features that must stay original whatever the product: brows, eyeballs,
             the mouth opening, and the lips or eyes when the product is not for them. Nothing
             generated is used inside it.
    """

    makeup: np.ndarray
    skin: np.ndarray
    protect: np.ndarray

    @property
    def edit(self) -> np.ndarray:
        """Everything the image model may change: makeup and skin, minus protected features."""
        return np.maximum(self.makeup, self.skin) * (self.protect < 0.5)


def _poly(mask: np.ndarray, points: np.ndarray, value: float) -> None:
    cv2.fillPoly(mask, [points.round().astype(np.int32)], value)


def _shape(points: np.ndarray, shape: tuple[int, int], polygons, margin: float) -> np.ndarray:
    """Filled polygons, grown by margin pixels."""
    mask = np.zeros(shape, dtype=np.float32)
    for ids in polygons:
        _poly(mask, points[ids], 1.0)
    return grow(mask, round(margin))


def region_masks(points: np.ndarray, shape: tuple[int, int], regions: set[str]) -> RegionMasks:
    unknown = set(regions) - set(REGIONS)
    if unknown:
        raise ValueError(f"unknown regions: {sorted(unknown)}")
    face_width = float(np.linalg.norm(points[FACE_EDGES[0]] - points[FACE_EDGES[1]]))

    makeup = np.zeros(shape, dtype=np.float32)
    if "cheeks" in regions:
        axes = (int(face_width * 0.13), int(face_width * 0.10))
        for index in CHEEKS:
            cv2.ellipse(makeup, tuple(points[index].round().astype(int)), axes, 0, 0, 360, 1.0, -1)
    if "eyes" in regions:
        for ids in EYE_SOCKETS:
            _poly(makeup, points[ids], 1.0)
    if "lips" in regions:
        _poly(makeup, points[LIPS_OUTER], 1.0)

    skin = np.zeros(shape, dtype=np.float32)
    if "skin" in regions:
        _poly(skin, points[FACE_OVAL], 1.0)

    # Margins are generous on purpose: a model that redraws the face moves brows and lip edges
    # by a few percent of the face width, and those moved copies must never be pasted.
    # Eye products sit right under the brow, so the brow margin is smaller for them.
    brow_margin = face_width * (0.012 if "eyes" in regions else 0.04)
    protected = [
        _shape(points, shape, BROWS, brow_margin),
        _shape(points, shape, EYE_OPENINGS, face_width * 0.006),
        _shape(points, shape, [LIPS_INNER], face_width * 0.004),
    ]
    if "lips" not in regions:
        protected.append(_shape(points, shape, [LIPS_OUTER], face_width * 0.035))
    if "eyes" not in regions:
        protected.append(_shape(points, shape, EYE_SOCKETS, face_width * 0.02))
    protect = feather(np.maximum.reduce(protected), face_width * 0.01)

    return RegionMasks(makeup * (protect < 0.5), skin * (protect < 0.5), protect)


def feather(mask: np.ndarray, radius: float) -> np.ndarray:
    """Soft edged version of a binary mask, for compositing without visible seams."""
    k = max(3, int(radius) * 2 + 1)
    return np.clip(cv2.GaussianBlur(mask, (k, k), 0), 0.0, 1.0)


def grow(mask: np.ndarray, pixels: int) -> np.ndarray:
    """Expands a binary mask a little, so the model has room to blend at the edges."""
    if pixels <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    return cv2.dilate(mask, kernel)
