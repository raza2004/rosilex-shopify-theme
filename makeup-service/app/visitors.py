"""Visitor log for the try on service, saved as JSON (VISITORS_FILE).

One entry per IP address: approximate city and country, first and last visit, how many times
the service was used, and each use (time, product, outcome, seconds on site). Photos are never
stored here. City and country come from an offline database, so IPs are not sent anywhere.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

MAX_EVENTS_PER_VISITOR = 200


class GeoLookup:
    """City and country for an IP from a local DB-IP Lite (or MaxMind) .mmdb file."""

    def __init__(self, path: str):
        self._reader = None
        if path and Path(path).exists():
            import maxminddb

            self._reader = maxminddb.open_database(path)

    def locate(self, ip: str) -> tuple[str | None, str | None]:
        if self._reader is None:
            return None, None
        try:
            record = self._reader.get(ip) or {}
        except ValueError:  # not an IP address
            return None, None
        city = (record.get("city") or {}).get("names", {}).get("en")
        country = (record.get("country") or {}).get("names", {}).get("en")
        return city, country


class VisitorLog:
    def __init__(self, path: str, geo: GeoLookup):
        self.path = Path(path)
        self.geo = geo
        self._lock = threading.Lock()

    def _load(self) -> dict:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8") or "{}")
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def snapshot(self) -> dict:
        with self._lock:
            return self._load()

    def record(self, ip: str, product: str, outcome: str, seconds_on_site: int | None) -> None:
        """Adds one use. outcome: passed, failed, error or daily_limit."""
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            data = self._load()
            visitor = data.get(ip)
            if visitor is None:
                city, country = self.geo.locate(ip)
                visitor = data[ip] = {"ip": ip, "city": city, "country": country, "first_seen": now, "uses": 0, "events": []}
            visitor["last_seen"] = now
            if outcome != "daily_limit":
                visitor["uses"] += 1
            if seconds_on_site is not None:
                visitor["seconds_on_site"] = max(seconds_on_site, visitor.get("seconds_on_site") or 0)
            visitor["events"] = (visitor["events"] + [
                {"time": now, "product": product, "outcome": outcome, "seconds_on_site": seconds_on_site}
            ])[-MAX_EVENTS_PER_VISITOR:]
            # Write in place: the file may be a single file bind mount, where renaming over it fails.
            with open(self.path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
                file.write("\n")
