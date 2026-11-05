"""Runtime settings, read once from environment variables and config/threshold.json."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parent.parent
THRESHOLD_FILE = SERVICE_ROOT / "config" / "threshold.json"

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ATTEMPTS = 3


def _csv(value: str) -> list[str]:
    return [item.strip().rstrip("/") for item in value.split(",") if item.strip()]


def load_threshold(path: Path = THRESHOLD_FILE) -> float:
    """The pass threshold for identity similarity, as chosen by eval/calibrate.py."""
    data = json.loads(path.read_text(encoding="utf-8"))
    threshold = float(data["threshold"])
    if not -1.0 <= threshold <= 1.0:
        raise ValueError(f"threshold {threshold} in {path} is not a cosine similarity")
    return threshold


@dataclass(frozen=True)
class Settings:
    generator: str = "gemini"  # or "openai"; "local" is the free evaluation baseline
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash-image"
    openai_api_key: str = ""
    openai_model: str = "gpt-image-1.5"  # gpt-image-1 was shut down on 2026-10-23
    allowed_origins: list[str] = field(default_factory=list)
    threshold: float = 0.0
    max_attempts: int = MAX_ATTEMPTS
    max_upload_bytes: int = MAX_UPLOAD_BYTES
    daily_generation_limit: int = 3  # image model calls per IP per 24 hours; 0 turns the limit off
    quota_exempt_ips: list[str] = field(default_factory=list)
    visitors_file: str = ""  # path of the visitor log (JSON); empty turns logging off
    geoip_db: str = "/app/geo/dbip-city-lite.mmdb"
    admin_token: str = ""  # lets the owner read the visitor log at GET /admin/visitors
    debug_save_dir: str = ""  # saves inputs and raw model output, for exempt (owner) IPs only
    rate_limit_requests: int = 10
    rate_limit_window_seconds: int = 600
    trust_forwarded_for: bool = False
    model_root: str = str(SERVICE_ROOT / "models")

    @classmethod
    def from_env(cls) -> "Settings":
        threshold_env = os.environ.get("IDENTITY_THRESHOLD")
        generator = os.environ.get("GENERATOR", cls.generator).strip().lower()
        if generator not in {"gemini", "openai", "local"}:
            raise ValueError(f'GENERATOR must be "gemini", "openai" or "local", not "{generator}"')
        return cls(
            generator=generator,
            gemini_api_key=os.environ.get("GEMINI_API_KEY", ""),
            gemini_model=os.environ.get("GEMINI_IMAGE_MODEL", cls.gemini_model),
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            openai_model=os.environ.get("OPENAI_IMAGE_MODEL", cls.openai_model),
            allowed_origins=_csv(os.environ.get("ALLOWED_ORIGINS", "")),
            threshold=float(threshold_env) if threshold_env else load_threshold(),
            max_attempts=min(MAX_ATTEMPTS, max(1, int(os.environ.get("MAX_ATTEMPTS", MAX_ATTEMPTS)))),
            daily_generation_limit=int(os.environ.get("DAILY_GENERATION_LIMIT", cls.daily_generation_limit)),
            quota_exempt_ips=_csv(os.environ.get("QUOTA_EXEMPT_IPS", "")),
            visitors_file=os.environ.get("VISITORS_FILE", ""),
            geoip_db=os.environ.get("GEOIP_DB", cls.geoip_db),
            admin_token=os.environ.get("ADMIN_TOKEN", ""),
            debug_save_dir=os.environ.get("DEBUG_SAVE_DIR", ""),
            rate_limit_requests=int(os.environ.get("RATE_LIMIT_REQUESTS", cls.rate_limit_requests)),
            rate_limit_window_seconds=int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", cls.rate_limit_window_seconds)),
            trust_forwarded_for=os.environ.get("TRUST_FORWARDED_FOR", "").lower() in {"1", "true", "yes"},
            model_root=os.environ.get("INSIGHTFACE_ROOT", cls.model_root),
        )
