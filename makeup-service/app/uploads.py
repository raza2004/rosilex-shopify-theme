"""Upload validation. Everything stays in memory: nothing here touches the disk."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# Longest side sent to Gemini and used for the identity check. Larger photos add cost, not detail.
WORK_SIZE = 1536

ALLOWED_FORMATS = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}


class UploadError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def sniff_format(data: bytes) -> str | None:
    """Detects the real file type from its first bytes, ignoring the declared content type."""
    if data.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "WEBP"
    return None


def load_image(data: bytes, max_bytes: int) -> Image.Image:
    """Checks size and type, decodes, applies EXIF rotation and returns an RGB image (EXIF dropped)."""
    if not data:
        raise UploadError(400, "empty_file", "The uploaded file is empty.")
    if len(data) > max_bytes:
        raise UploadError(413, "file_too_large", f"The image must be {max_bytes // (1024 * 1024)} MB or smaller.")
    if sniff_format(data) is None:
        raise UploadError(415, "unsupported_type", "Upload a JPEG, PNG or WebP image.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise UploadError(415, "unreadable_image", "The image could not be read.") from error
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((WORK_SIZE, WORK_SIZE))
    return image


def to_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def to_bgr(image: Image.Image) -> np.ndarray:
    """InsightFace expects OpenCV channel order."""
    return np.asarray(image.convert("RGB"))[:, :, ::-1].copy()
