import base64
import io
import tempfile

import numpy as np
import pytest
from PIL import Image

from app.generator import GenerationError
from app.styles import STYLES

from .conftest import NO_FACE, THRESHOLD, TWO_FACES, FakeGenerator, FakeRegions, solid


def post(client, upload, **kwargs):
    return client.post("/api/makeup", **upload(**kwargs))


# Styles


def test_styles_lists_the_five_looks(client_for):
    body = client_for(FakeGenerator()).get("/api/styles").json()
    assert [s["id"] for s in body["styles"]] == ["natural", "soft_glam", "bold_lip", "smoky_eye", "bridal"]
    assert all(s["name"] and s["prompt"] for s in body["styles"])


def test_prompt_asks_to_keep_identity(client_for, upload):
    generator = FakeGenerator(0.9)
    post(client_for(generator), upload, style="smoky_eye")
    prompt = generator.prompts[0]
    for kept in ("face shape", "skin texture", "expression", "pose", "hair", "background", "clothing"):
        assert kept in prompt
    assert "makeup only" in prompt
    assert "smoky eye" in prompt


# Upload validation


def test_unknown_style_is_rejected(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, style="glitter")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unknown_style"


def test_product_is_applied_only_to_its_regions(client_for, upload):
    generator, regions = FakeGenerator(0.9), FakeRegions()
    product = {"title": "Velvet lipstick", "type": "Lipstick", "vendor": "Rosilex", "shade": "Ruby", "description": "<p>Matte red.</p>"}
    body = post(client_for(generator, regions=regions), upload, style=None, product=product).json()
    assert regions.requested == [{"lips"}]
    assert body["regions"] == ["lips"] and body["name"] == "Velvet lipstick" and body["look"] == "product"
    prompt = generator.prompts[0]
    assert "Velvet lipstick" in prompt and "Ruby" in prompt and "Matte red." in prompt and "<p>" not in prompt
    assert "only to the lips" in prompt


def test_skincare_product_edits_the_skin(client_for, upload):
    regions = FakeRegions()
    post(client_for(FakeGenerator(0.9), regions=regions), upload, style=None, product={"title": "Daily serum", "type": "Serum"})
    assert regions.requested == [{"skin"}]


@pytest.mark.parametrize("product", ["not json", "[1, 2]", '{"type": "Lipstick"}'])
def test_invalid_product_is_rejected(client_for, upload, product):
    response = post(client_for(FakeGenerator()), upload, style=None, product=product)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_product"


def test_photo_without_landmarks_is_rejected(client_for, upload):
    generator = FakeGenerator(0.9)
    response = post(client_for(generator, regions=FakeRegions(landmarks=False)), upload)
    assert response.status_code == 422 and generator.calls == 0


def test_model_receives_mask_and_standard_size(client_for, upload):
    generator = FakeGenerator(0.9)
    post(client_for(generator), upload)
    request = generator.requests[0]
    assert request.size == (1024, 1024)
    mask = Image.open(io.BytesIO(request.mask_png))
    assert mask.mode == "RGBA" and mask.size == request.size


def test_missing_image_is_rejected(client_for):
    response = client_for(FakeGenerator()).post("/api/makeup", data={"style": "natural"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_non_image_is_rejected_even_with_image_content_type(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, data=b"%PDF-1.7 not a photo")
    assert response.status_code == 415
    assert response.json()["error"]["code"] == "unsupported_type"


def test_truncated_image_is_rejected(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, data=solid(120, fmt="JPEG")[:40])
    assert response.status_code == 415


def test_empty_file_is_rejected(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, data=b"")
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "empty_file"


def test_file_over_8_mb_is_rejected(client_for, upload):
    big = b"\xff\xd8\xff" + b"0" * (8 * 1024 * 1024 + 10)
    response = post(client_for(FakeGenerator()), upload, data=big)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_body_far_over_the_limit_is_stopped_before_parsing(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, data=b"\xff\xd8\xff" + b"0" * (12 * 1024 * 1024))
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_photo_without_a_face_is_rejected(client_for, upload):
    generator = FakeGenerator()
    response = post(client_for(generator), upload, data=solid(NO_FACE, fmt="JPEG"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "no_face"
    assert generator.calls == 0


def test_photo_with_two_faces_is_rejected(client_for, upload):
    response = post(client_for(FakeGenerator()), upload, data=solid(TWO_FACES, fmt="JPEG"))
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "multiple_faces"


@pytest.mark.parametrize("fmt", ["PNG", "WEBP"])
def test_png_and_webp_are_accepted(client_for, upload, fmt):
    response = post(client_for(FakeGenerator(0.9)), upload, data=solid(120, fmt=fmt))
    assert response.status_code == 200


def test_large_upload_stays_in_memory(client_for, upload, monkeypatch):
    """A ~7 MB upload must not roll over from memory to a temporary file on disk."""

    def fail_rollover(self):
        raise AssertionError("upload was spooled to disk")

    monkeypatch.setattr(tempfile.SpooledTemporaryFile, "rollover", fail_rollover)
    noise = np.random.default_rng(1).integers(0, 255, (1400, 1700, 3), dtype=np.uint8)
    noise[..., 0], noise[..., 1] = 120, 0  # one face, green 0, for the fake detector; blue stays noise
    buffer = io.BytesIO()
    Image.fromarray(noise).save(buffer, format="PNG", compress_level=0)
    assert 6 * 1024 * 1024 < len(buffer.getvalue()) < 8 * 1024 * 1024
    response = post(client_for(FakeGenerator(0.9)), upload, data=buffer.getvalue(), filename="me.png")
    assert response.status_code == 200


# Generation and retries


def test_first_attempt_that_passes_is_returned(client_for, upload):
    generator = FakeGenerator(0.92, 0.99)
    body = post(client_for(generator), upload).json()
    assert body["attempts"] == 1 and generator.calls == 1
    assert body["passed"] is True
    assert body["identity_score"] == pytest.approx(0.92, abs=0.01)
    assert body["threshold"] == THRESHOLD
    assert Image.open(io.BytesIO(base64.b64decode(body["image"]))).size == (64, 64)


def test_retries_after_low_score_and_missing_face(client_for, upload):
    generator = FakeGenerator(0.2, None, 0.8)
    body = post(client_for(generator), upload).json()
    assert body["attempts"] == 3
    assert body["passed"] is True
    assert body["identity_score"] == pytest.approx(0.8, abs=0.01)


def test_best_attempt_is_returned_when_none_pass(client_for, upload):
    body = post(client_for(FakeGenerator(0.2, 0.45, 0.3)), upload).json()
    assert body["attempts"] == 3
    assert body["passed"] is False
    assert body["identity_score"] == pytest.approx(0.45, abs=0.01)


def test_no_more_than_three_attempts(client_for, upload):
    generator = FakeGenerator(0.1, 0.1, 0.1, 0.99)
    body = post(client_for(generator), upload).json()
    assert generator.calls == 3 and body["attempts"] == 3


def test_result_without_any_face_is_reported_as_failed(client_for, upload):
    body = post(client_for(FakeGenerator(None, None, None)), upload).json()
    assert body["passed"] is False
    assert body["face_detected"] is False
    assert body["identity_score"] is None
    assert body["image"]


def test_gemini_failure_is_retried(client_for, upload):
    body = post(client_for(FakeGenerator(RuntimeError("503"), 0.9)), upload).json()
    assert body["attempts"] == 2 and body["passed"] is True


def test_gemini_failing_every_time_returns_502(client_for, upload):
    response = post(client_for(FakeGenerator(*[RuntimeError("503")] * 3)), upload)
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "generation_failed"


def test_quota_error_is_not_retried_and_returns_503(client_for, upload):
    generator = FakeGenerator(GenerationError("Gemini HTTP 429", retryable=False, quota=True), 0.9, 0.9)
    response = post(client_for(generator), upload)
    assert generator.calls == 1
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "generation_busy"
    assert response.headers["retry-after"] == "60"


def test_invalid_request_error_stops_retrying(client_for, upload):
    generator = FakeGenerator(GenerationError("Gemini HTTP 400", retryable=False), 0.9)
    response = post(client_for(generator), upload)
    assert generator.calls == 1
    assert response.status_code == 502


def test_missing_api_key_returns_503(client_for, upload):
    response = post(client_for(None, gemini_api_key=""), upload)
    assert response.status_code == 503


# CORS and rate limit


def test_cors_allows_the_store_only(client_for):
    client = client_for(FakeGenerator())
    allowed = client.get("/api/styles", headers={"Origin": "https://rosexious.myshopify.com"})
    other = client.get("/api/styles", headers={"Origin": "https://example.com"})
    assert allowed.headers["access-control-allow-origin"] == "https://rosexious.myshopify.com"
    assert "access-control-allow-origin" not in other.headers


def test_rate_limit_returns_429_with_retry_after(client_for, upload):
    client = client_for(FakeGenerator(0.9, 0.9), rate_limit_requests=2, rate_limit_window_seconds=60)
    assert post(client, upload).status_code == 200
    assert post(client, upload).status_code == 200
    response = post(client, upload)
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) > 0


def test_every_style_can_be_requested(client_for, upload):
    for style in STYLES:
        response = post(client_for(FakeGenerator(0.9)), upload, style=style.id)
        assert response.status_code == 200 and response.json()["look"] == style.id
