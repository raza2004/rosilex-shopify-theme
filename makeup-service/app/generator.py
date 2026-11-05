"""Image engines, chosen with the GENERATOR env var.

openai: OpenAI image edits (gpt-image-1.5 by default) with an edit mask and high input fidelity.
local:  app.local_makeup, a free baseline that paints preset styles (evaluation only, not generative).
gemini: Gemini 2.5 Flash Image. It takes no mask; compositing still keeps everything outside
        the makeup regions original.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Protocol

# Models whose image edit endpoint accepts input_fidelity="high" (OpenAI image edit reference).
HIGH_FIDELITY_MODELS = ("gpt-image-1.5",)


@dataclass(frozen=True)
class EditRequest:
    image_png: bytes  # photo padded to a model size
    mask_png: bytes  # same size, transparent where editing is allowed
    size: tuple[int, int]
    prompt: str
    look_id: str = ""  # preset style id or "product"; the local baseline engine needs it


@dataclass(frozen=True)
class GeneratedImage:
    data: bytes
    mime_type: str
    usage: dict | None = None  # token usage reported by the API, when it reports it


class GenerationError(Exception):
    """The engine produced no image: the call failed, was refused, or returned no image.

    retryable is False when trying again right away cannot help: quota or rate limit (429)
    and invalid requests (other 4xx).
    """

    def __init__(self, message: str, retryable: bool = True, quota: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.quota = quota


class ImageGenerator(Protocol):
    engine: str

    def generate(self, request: EditRequest) -> GeneratedImage: ...


class OpenAIGenerator:
    engine = "openai"

    def __init__(self, api_key: str, model: str):
        from openai import OpenAI

        if not api_key:
            raise GenerationError("OPENAI_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, timeout=120, max_retries=0)
        self._model = model

    def generate(self, request: EditRequest) -> GeneratedImage:
        import openai

        options = {}
        if self._model.startswith(HIGH_FIDELITY_MODELS):
            options["input_fidelity"] = "high"  # keep faces and details from the input
        try:
            response = self._client.images.edit(
                model=self._model,
                image=("photo.png", request.image_png, "image/png"),
                mask=("mask.png", request.mask_png, "image/png"),
                prompt=request.prompt,
                size=f"{request.size[0]}x{request.size[1]}",
                quality="medium",
                **options,
            )
        except openai.RateLimitError as error:
            raise GenerationError("OpenAI HTTP 429", retryable=False, quota=True) from error
        except openai.APIStatusError as error:
            code = error.status_code
            raise GenerationError(f"OpenAI HTTP {code}", retryable=code >= 500) from error
        except openai.APIError as error:  # connection errors and timeouts: worth another attempt
            raise GenerationError(f"OpenAI request failed: {type(error).__name__}") from error

        data = response.data[0].b64_json if response.data else None
        if not data:
            raise GenerationError("OpenAI returned no image")
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return GeneratedImage(base64.b64decode(data), "image/png", usage)


class GeminiGenerator:
    engine = "gemini"

    def __init__(self, api_key: str, model: str):
        from google import genai

        if not api_key:
            raise GenerationError("GEMINI_API_KEY is not set")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate(self, request: EditRequest) -> GeneratedImage:
        from google.genai import errors, types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=[types.Part.from_bytes(data=request.image_png, mime_type="image/png"), request.prompt],
                config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
            )
        except errors.APIError as error:
            code = getattr(error, "code", None) or 0
            raise GenerationError(
                f"Gemini HTTP {code}", retryable=code >= 500 or code == 0, quota=code == 429
            ) from error
        except Exception as error:  # network errors and timeouts: worth another attempt
            raise GenerationError(f"Gemini request failed: {type(error).__name__}") from error

        for candidate in response.candidates or []:
            for part in (candidate.content.parts if candidate.content else None) or []:
                if part.inline_data and part.inline_data.data:
                    return GeneratedImage(part.inline_data.data, part.inline_data.mime_type or "image/png")

        feedback = response.prompt_feedback
        reason = feedback.block_reason if feedback and feedback.block_reason else None
        if reason is None and response.candidates:
            reason = response.candidates[0].finish_reason
        raise GenerationError(f"Gemini returned no image ({reason or 'unknown reason'})")
