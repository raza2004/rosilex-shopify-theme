"""Prepares the photo and edit mask for the image model, and composites its result back.

The photo is padded (edge pixels repeated) to the nearest standard model size so nothing is
stretched. Afterwards the result is combined with the original photo:

* makeup regions (lips, eyes, cheeks): generated pixels are pasted with a soft edge, inside the
  original feature's outline, so a lipstick cannot make the lips bigger;
* skin: the generated skin is used, so glow, sheen and finish show, except wherever it is clearly
  darker than the original. Skincare brightens, evens and adds glow but never adds dark features,
  so dark changes are brows, lip edges or shadows the model moved: there the original is kept, and
  they cannot appear twice;
* protected features (brows, eyeballs, mouth) and everything outside the masks: original pixels.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from .regions import RegionMasks, feather, grow

# Sizes every GPT image model accepts (OpenAI image edit reference).
MODEL_SIZES = ((1024, 1024), (1536, 1024), (1024, 1536))


@dataclass(frozen=True)
class Prepared:
    image_png: bytes  # padded photo at model size
    mask_png: bytes  # same size; transparent where the model may edit
    size: tuple[int, int]  # model size (width, height)
    box: tuple[int, int, int, int]  # where the original photo sits inside it (left, top, right, bottom)
    makeup_weight: np.ndarray  # 0..1 at the original resolution: paste generated pixels
    skin_weight: np.ndarray  # 0..1 at the original resolution: use generated skin, unless darkened
    guard_radius: int  # how far around a darkened spot the original is kept, in pixels


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def model_size(width: int, height: int) -> tuple[int, int]:
    aspect = width / height
    return min(MODEL_SIZES, key=lambda s: abs(math.log((s[0] / s[1]) / aspect)))


def prepare(original: Image.Image, masks: RegionMasks) -> Prepared:
    width, height = original.size
    short_side = min(width, height)
    target_w, target_h = model_size(width, height)

    # Pad to the model's aspect ratio, centred.
    padded_w = max(width, round(height * target_w / target_h))
    padded_h = max(height, round(width * target_h / target_w))
    left, top = (padded_w - width) // 2, (padded_h - height) // 2
    pads = (top, padded_h - height - top, left, padded_w - width - left)

    rgb = cv2.copyMakeBorder(np.asarray(original.convert("RGB")), *pads, cv2.BORDER_REPLICATE)
    # A little room to blend at the edges, but never over protected features.
    edit = grow(masks.edit, max(2, round(short_side * 0.01))) * (masks.protect < 0.5)
    edit = cv2.copyMakeBorder(edit.astype(np.float32), *pads, cv2.BORDER_CONSTANT, value=0)

    image = Image.fromarray(rgb).resize((target_w, target_h), Image.LANCZOS)
    edit = cv2.resize(edit, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    alpha = np.where(edit > 0.5, 0, 255).astype(np.uint8)  # transparent = editable
    mask = Image.fromarray(np.dstack([np.zeros((target_h, target_w, 3), np.uint8), alpha]), "RGBA")

    scale = target_w / padded_w
    box = (round(left * scale), round(top * scale), round((left + width) * scale), round((top + height) * scale))
    keep = 1.0 - masks.protect
    makeup = feather(masks.makeup, short_side * 0.006) * keep
    skin = feather(masks.skin, short_side * 0.02) * keep * (1.0 - makeup)
    return Prepared(_png(image), _png(mask), (target_w, target_h), box, makeup, skin, max(2, round(short_side * 0.008)))


# Darkening (in 0..255 brightness) that skincare can cause, such as a matte finish, before the
# original is kept instead; fully kept from DARKEN_FULL on.
DARKEN_START = 14
DARKEN_FULL = 34


def darkened(base: np.ndarray, made: np.ndarray, radius: int) -> np.ndarray:
    """0..1 where the generated image is clearly darker than the original, grown and softened."""
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    drop = base @ weights - made @ weights
    guard = np.clip((drop - DARKEN_START) / (DARKEN_FULL - DARKEN_START), 0, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    guard = cv2.dilate(guard, kernel)  # also cover the soft fringe of a moved brow or lip edge
    return np.clip(cv2.GaussianBlur(guard, (0, 0), sigmaX=radius), 0, 1)


def composite(original: Image.Image, generated: bytes, prepared: Prepared) -> Image.Image:
    """The original photo with only the allowed changes taken from the generated image."""
    result = Image.open(io.BytesIO(generated)).convert("RGB")
    if result.size != prepared.size:
        result = result.resize(prepared.size, Image.LANCZOS)
    result = result.crop(prepared.box).resize(original.size, Image.LANCZOS)

    base = np.asarray(original.convert("RGB"), dtype=np.float32)
    made = np.asarray(result, dtype=np.float32)
    weight = prepared.makeup_weight
    if prepared.skin_weight.any():
        skin = prepared.skin_weight * (1 - darkened(base, made, prepared.guard_radius))
        weight = np.maximum(weight, skin)
    weight = weight[..., None]
    out = base * (1 - weight) + made * weight
    return Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8))
