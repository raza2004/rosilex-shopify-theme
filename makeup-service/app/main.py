"""FastAPI app for the makeup demo. Start with: uvicorn app.main:create_app --factory"""

from __future__ import annotations

import base64
import json
import logging
import secrets

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.formparsers import MultiPartParser

from . import pipeline
from .config import Settings
from .generator import GeminiGenerator, ImageGenerator, OpenAIGenerator
from .identity import FaceEmbedder, InsightFaceEmbedder
from .quota import DailyQuota
from .ratelimit import RateLimiter
from .regions import FaceRegions, NoLandmarksError
from .styles import STYLES, STYLES_BY_ID, look_for_product, look_for_style, parse_product
from .uploads import UploadError, load_image, to_bgr
from .visitors import GeoLookup, VisitorLog

logger = logging.getLogger("makeup_service")

# Multipart overhead allowed on top of the image itself.
FORM_OVERHEAD_BYTES = 64 * 1024


def error_response(status: int, code: str, message: str, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


class BodySizeLimit:
    """Rejects request bodies over the limit while they stream in, before anything is parsed or spooled."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        headers = dict(scope.get("headers") or [])
        declared = headers.get(b"content-length")
        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            return await self._reject(scope, receive, send)

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await self._reject(scope, receive, send)

    async def _reject(self, scope, receive, send):
        mb = (self.max_bytes - FORM_OVERHEAD_BYTES) // (1024 * 1024)
        response = error_response(413, "file_too_large", f"The image must be {mb} MB or smaller.")
        await response(scope, receive, send)


class _TooLarge(HTTPException):
    # An HTTPException, because FastAPI rewraps any other error raised during form parsing as a 400.
    def __init__(self):
        super().__init__(status_code=413)


def client_key(request: Request, trust_forwarded_for: bool) -> str:
    if trust_forwarded_for:
        forwarded = request.headers.get("x-forwarded-for", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _product_title(product: str | None) -> str | None:
    """The product title from the raw form field, for the visitor log, without failing."""
    try:
        return str(json.loads(product).get("title") or "")[:150] if product else None
    except (ValueError, AttributeError):
        return None


def save_debug(folder: str, photo, look, masks, result) -> None:
    """Owner debugging only: keeps the inputs and every raw model output, so compositing can be
    tuned offline without paying for new generations. Never used for visitors' requests."""
    import numpy as np
    from datetime import datetime
    from pathlib import Path

    try:
        out = Path(folder) / datetime.now().strftime("%Y%m%d-%H%M%S")
        out.mkdir(parents=True, exist_ok=True)
        photo.save(out / "original.png")
        (out / "prompt.txt").write_text(
            f"{look.name}\nregions: {sorted(look.regions)}\n\n{look.prompt}\n", encoding="utf-8"
        )
        np.savez_compressed(out / "masks.npz", makeup=masks.makeup, skin=masks.skin, protect=masks.protect)
        for attempt in result.attempts:
            if attempt.raw:
                (out / f"raw_{attempt.number}.png").write_bytes(attempt.raw)
            if attempt.image:
                (out / f"final_{attempt.number}_score_{attempt.score}.png").write_bytes(attempt.image)
    except OSError:
        logger.warning("could not save debug output")


def build_generator(settings: Settings) -> ImageGenerator | None:
    """The engine named by GENERATOR, or None when its API key is not set."""
    if settings.generator == "local":
        from .local_makeup import LocalMakeupGenerator

        return LocalMakeupGenerator()
    if settings.generator == "openai":
        return OpenAIGenerator(settings.openai_api_key, settings.openai_model) if settings.openai_api_key else None
    if settings.gemini_api_key:
        return GeminiGenerator(settings.gemini_api_key, settings.gemini_model)
    return None


def create_app(
    settings: Settings | None = None,
    generator: ImageGenerator | None = None,
    embedder: FaceEmbedder | None = None,
    regions: FaceRegions | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    embedder = embedder or InsightFaceEmbedder(settings.model_root)
    regions = regions or FaceRegions()
    if generator is None:
        generator = build_generator(settings)

    # Keep multipart uploads in memory: Starlette spools parts over 1 MB to a temp file by default.
    # The setting is max_file_size in older Starlette releases and spool_max_size in newer ones.
    spool_attr = "spool_max_size" if hasattr(MultiPartParser, "spool_max_size") else "max_file_size"
    setattr(MultiPartParser, spool_attr, settings.max_upload_bytes + FORM_OVERHEAD_BYTES)

    limiter = RateLimiter(settings.rate_limit_requests, settings.rate_limit_window_seconds)
    visitors = VisitorLog(settings.visitors_file, GeoLookup(settings.geoip_db)) if settings.visitors_file else None
    quota = (
        DailyQuota(settings.daily_generation_limit, exempt=settings.quota_exempt_ips)
        if settings.daily_generation_limit > 0
        else None
    )

    def daily_limit_response(ip: str) -> JSONResponse:
        return error_response(
            429, "daily_limit", f"You have used today's {settings.daily_generation_limit} free try ons. Please come back tomorrow.",
            {"Retry-After": str(quota.retry_after(ip))},
        )
    app = FastAPI(title="Rosilex makeup demo", docs_url=None, redoc_url=None)
    app.add_middleware(BodySizeLimit, max_bytes=settings.max_upload_bytes + FORM_OVERHEAD_BYTES)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=600,
    )

    @app.exception_handler(UploadError)
    async def upload_error(_: Request, error: UploadError):
        return error_response(error.status, error.code, error.message)

    @app.exception_handler(_TooLarge)
    async def too_large(_: Request, __: _TooLarge):
        mb = settings.max_upload_bytes // (1024 * 1024)
        return error_response(413, "file_too_large", f"The image must be {mb} MB or smaller.")

    @app.exception_handler(RequestValidationError)
    async def validation_error(_: Request, error: RequestValidationError):
        missing = sorted({str(e["loc"][-1]) for e in error.errors()})
        return error_response(400, "invalid_request", f"Missing or invalid fields: {', '.join(missing)}.")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/admin/visitors")
    def visitor_log(request: Request):
        """The visitor log, for the owner only: send Authorization: Bearer <ADMIN_TOKEN>."""
        given = request.headers.get("authorization", "").removeprefix("Bearer ").strip()
        if not settings.admin_token or not secrets.compare_digest(given, settings.admin_token):
            return error_response(401, "unauthorized", "A valid admin token is required.")
        if not visitors:
            return error_response(404, "no_visitor_log", "The visitor log is turned off (set VISITORS_FILE).")
        return visitors.snapshot()

    @app.get("/api/styles")
    def styles():
        return {
            "engine": generator.engine if generator else None,
            "styles": [{"id": s.id, "name": s.name, "prompt": s.look, "regions": sorted(s.regions)} for s in STYLES],
        }

    @app.post("/api/makeup")
    async def makeup(
        request: Request,
        image: UploadFile = File(...),
        style: str | None = Form(None),
        product: str | None = Form(None),
        seconds_on_site: int | None = Form(None),
    ):
        ip = client_key(request, settings.trust_forwarded_for)
        seconds = None if seconds_on_site is None else max(0, min(seconds_on_site, 7 * 24 * 3600))

        def log(name: str, outcome: str) -> None:
            if visitors:
                try:
                    visitors.record(ip, name, outcome, seconds)
                except OSError:
                    logger.warning("could not write the visitor log")

        if quota and quota.remaining(ip) == 0:
            log(_product_title(product) or style or "", "daily_limit")
            return daily_limit_response(ip)

        retry_after = limiter.check(ip)
        if retry_after is not None:
            return error_response(
                429, "rate_limited", "Too many requests. Please wait a moment and try again.",
                {"Retry-After": str(retry_after)},
            )

        if product:
            try:
                look = look_for_product(parse_product(json.loads(product)))
            except (ValueError, TypeError):  # json.JSONDecodeError is a ValueError
                return error_response(400, "invalid_product", "The product details are missing or invalid.")
        elif style in STYLES_BY_ID:
            look = look_for_style(STYLES_BY_ID[style])
        else:
            return error_response(400, "unknown_style", "Send a product, or one of the styles from /api/styles.")

        data = await image.read(settings.max_upload_bytes + 1)
        await image.close()
        photo = load_image(data, settings.max_upload_bytes)
        del data

        faces = await run_in_threadpool(embedder.faces, to_bgr(photo))
        if not faces:
            return error_response(422, "no_face", "No face was found. Use a clear, front facing photo.")
        if len(faces) > 1:
            return error_response(422, "multiple_faces", "More than one face was found. Use a photo of one person.")

        try:
            masks = await run_in_threadpool(regions.masks, photo, set(look.regions))
        except NoLandmarksError:
            return error_response(422, "no_face", "No face was found. Use a clear, front facing photo.")

        if generator is None:
            return error_response(503, "not_configured", "Image generation is not configured on the server.")

        granted = quota.reserve(ip, settings.max_attempts) if quota else None
        if granted == 0:
            log(look.name, "daily_limit")
            return daily_limit_response(ip)
        attempts = settings.max_attempts if granted is None else granted
        used = 0
        try:
            result = await run_in_threadpool(
                pipeline.run, photo, faces[0].embedding, look, masks, generator, embedder,
                settings.threshold, attempts,
            )
            # Calls the provider refused with a quota error (429) are not billed, so they do not count.
            used = len(result.attempts) - (1 if result.quota_exceeded else 0)
        finally:
            if quota and granted:
                quota.refund(ip, granted - used)
        best = result.best
        if settings.debug_save_dir and ip in settings.quota_exempt_ips:
            save_debug(settings.debug_save_dir, photo, look, masks, result)
        if best is None or best.image is None:
            log(look.name, "error")
        else:
            log(look.name, "passed" if result.passed else "failed")
        if (best is None or best.image is None) and result.quota_exceeded:
            logger.warning("image model quota or rate limit reached")
            return error_response(
                503, "generation_busy", "The makeup service is busy right now. Please try again later.",
                {"Retry-After": "60"},
            )
        if best is None or best.image is None:
            logger.warning("generation failed after %d attempts", len(result.attempts))
            return error_response(502, "generation_failed", "The makeup could not be generated. Please try again.")

        return {
            "engine": generator.engine,
            "look": look.id,
            "name": look.name,
            "regions": sorted(look.regions),
            "image": base64.b64encode(best.image).decode("ascii"),
            "mime_type": best.mime_type,
            "identity_score": None if best.score is None else round(best.score, 4),
            "threshold": settings.threshold,
            "passed": result.passed,
            "face_detected": best.face_detected,
            "attempts": len(result.attempts),
            "max_attempts": settings.max_attempts,
            "tries_left": quota.remaining(ip) if quota else None,
        }

    return app
