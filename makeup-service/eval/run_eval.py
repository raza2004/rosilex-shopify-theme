"""Runs the makeup pipeline on the photos in eval/faces/ with preset styles, through the same code the API uses.

Results are kept per engine: eval/results/results_<engine>.csv and eval/results/outputs/<engine>/.
report.py turns them into eval/results/summary.md.

Paid engines (openai, gemini) are protected three ways:
* --max-calls is required and is a hard cap on API calls in this run; the run stops cleanly when it is reached.
* Every API result is cached in eval/results/cache/<engine>/, keyed by photo, mask, prompt and attempt
  number, so re-running anything already paid for costs nothing. A retry run reuses the cached first
  attempt and only pays for the extra attempts.
* Every API call is written to eval/results/calls_<engine>.jsonl with the token usage the API reports.

Examples:
    python eval/run_eval.py --generator local --modes single
    python eval/run_eval.py --generator openai --per-person 1 --limit 10 --styles soft_glam --modes single --max-calls 10
    python eval/run_eval.py --generator openai --per-person 1 --limit 10 --styles soft_glam --modes retry --only-failed --max-calls 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone

from common import RESULTS_DIR, embed_faces, list_faces, make_embedder

from app import pipeline
from app.config import MAX_ATTEMPTS, Settings
from app.generator import EditRequest, GeneratedImage, GenerationError
from app.main import build_generator
from app.regions import FaceRegions, NoLandmarksError
from app.styles import STYLES, STYLES_BY_ID, look_for_style

PAID_ENGINES = {"openai", "gemini"}
FIELDS = [
    "engine", "image", "person", "style", "mode", "score", "pass", "attempts", "api_calls", "cache_hits",
    "detection_failures", "generation_failures", "threshold", "seconds", "output", "note",
]
MODES = {"single": 1, "retry": MAX_ATTEMPTS}


CAP_MESSAGE = "paid call cap reached"


class MeteredGenerator:
    """Wraps an engine: counts calls, enforces the cap, caches results and logs API usage."""

    def __init__(self, inner, paid: bool, max_calls: int | None, cache_dir, ledger_path):
        self.inner = inner
        self.engine = inner.engine
        self.paid = paid
        self.max_calls = max_calls
        self.cache_dir = cache_dir
        self.ledger_path = ledger_path
        self.calls = 0  # real engine calls in this run
        self.cache_hits = 0
        self.cap_hit = False
        self._seen: dict[str, int] = defaultdict(int)

    def reset(self) -> None:
        """Start attempt numbering again for the next photo and style."""
        self._seen.clear()

    def generate(self, request: EditRequest) -> GeneratedImage:
        key = hashlib.sha256(
            self.engine.encode() + request.image_png + request.mask_png + request.prompt.encode()
        ).hexdigest()[:24]
        attempt = self._seen[key]
        self._seen[key] += 1
        cached = self.cache_dir / f"{key}_{attempt}.png" if self.cache_dir else None
        if cached and cached.exists():
            self.cache_hits += 1
            return GeneratedImage(cached.read_bytes(), "image/png")
        if self.paid and self.calls >= self.max_calls:
            # Stops this photo's retry loop but keeps the attempts already made, then the run ends.
            self.cap_hit = True
            raise GenerationError(CAP_MESSAGE, retryable=False)

        self.calls += 1
        entry = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "key": f"{key}_{attempt}"}
        try:
            result = self.inner.generate(request)
        except GenerationError as error:
            entry["error"] = str(error)
            self._log(entry)
            raise
        entry["usage"] = result.usage
        self._log(entry)
        if cached:
            cached.write_bytes(result.data)
        return result

    def _log(self, entry: dict) -> None:
        if self.paid:
            with open(self.ledger_path, "a", encoding="utf-8") as file:
                file.write(json.dumps(entry) + "\n")


def pick_faces(faces, per_person: int | None, limit: int | None):
    """Usable photos only when per_person is set: the first N per person, then the first `limit` of those."""
    if per_person:
        taken: dict[str, int] = defaultdict(int)
        chosen = []
        for face in faces:
            if face.embedding is not None and taken[face.person] < per_person:
                taken[face.person] += 1
                chosen.append(face)
        faces = chosen
    return faces[:limit] if limit else faces


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", choices=["local", "openai", "gemini"], help="engine (default: GENERATOR from the environment)")
    parser.add_argument("--modes", default="single,retry", help="single, retry or both, comma separated")
    parser.add_argument("--styles", help="comma separated style ids (default: all five)")
    parser.add_argument("--per-person", type=int, help="use only the first N usable photos of each person")
    parser.add_argument("--limit", type=int, help="only the first N photos (after --per-person: the first N people)")
    parser.add_argument("--max-calls", type=int, help="hard cap on paid API calls in this run (required for paid engines)")
    parser.add_argument("--only-failed", action="store_true", help="retry mode: only photos and styles that failed in single mode")
    parser.add_argument("--fresh", action="store_true", help="ignore existing rows for this engine")
    args = parser.parse_args()

    if args.generator:
        os.environ["GENERATOR"] = args.generator
    settings = Settings.from_env()
    engine_name = settings.generator
    paid = engine_name in PAID_ENGINES
    if paid and args.max_calls is None:
        raise SystemExit("A paid engine needs --max-calls, the most API calls this run may make.")
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if not set(modes) <= set(MODES):
        raise SystemExit(f"--modes must be from {sorted(MODES)}")
    styles = [STYLES_BY_ID[s] for s in args.styles.split(",")] if args.styles else list(STYLES)

    inner = build_generator(settings)
    if inner is None:
        raise SystemExit("Set the API key for the chosen engine (OPENAI_API_KEY or GEMINI_API_KEY).")
    results_csv = RESULTS_DIR / f"results_{engine_name}.csv"
    outputs_dir = RESULTS_DIR / "outputs" / engine_name
    cache_dir = RESULTS_DIR / "cache" / engine_name if paid else None
    for folder in (outputs_dir, cache_dir):
        if folder:
            folder.mkdir(parents=True, exist_ok=True)
    generator = MeteredGenerator(inner, paid, args.max_calls, cache_dir, RESULTS_DIR / f"calls_{engine_name}.jsonl")

    embedder = make_embedder()
    faces = pick_faces(embed_faces(list_faces(), embedder), args.per_person, args.limit)
    regions = FaceRegions()

    rows = [] if args.fresh or not results_csv.exists() else list(csv.DictReader(open(results_csv, newline="", encoding="utf-8")))
    done = {(r["image"], r["style"], r["mode"]) for r in rows}
    # Retrying helps low scores and missing faces, not requests the API refused (HTTP 4xx).
    failed_single = {
        (r["image"], r["style"]) for r in rows
        if r["mode"] == "single" and r["pass"] != "1" and r.get("attempts") not in ("0", "") and "HTTP 4" not in (r.get("note") or "")
    }
    print(f"Engine: {engine_name}. Photos: {len(faces)}. Styles: {', '.join(s.id for s in styles)}. Modes: {', '.join(modes)}."
          + (f" Paid call cap: {args.max_calls}." if paid else ""))

    stopped = False
    with open(results_csv, "w" if args.fresh or not results_csv.exists() else "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        if args.fresh or not rows:
            writer.writeheader()
        for face in faces:
            for style in styles:
                for mode in modes:
                    if (face.name, style.id, mode) in done:
                        continue
                    if mode == "retry" and args.only_failed and (face.name, style.id) not in failed_single:
                        continue
                    row = {"engine": engine_name, "image": face.name, "person": face.person, "style": style.id,
                           "mode": mode, "threshold": settings.threshold}
                    look = look_for_style(style)
                    masks = None
                    if face.embedding is not None:
                        try:
                            masks = regions.masks(face.image, set(look.regions))
                        except NoLandmarksError:
                            face.problem = "no_landmarks"
                    if masks is None:
                        row.update({"pass": 0, "attempts": 0, "api_calls": 0, "note": f"input skipped: {face.problem}"})
                    else:
                        calls_before, hits_before = generator.calls, generator.cache_hits
                        generator.reset()
                        started = time.monotonic()
                        result = pipeline.run(face.image, face.embedding, look, masks, generator, embedder,
                                              settings.threshold, MODES[mode])
                        if generator.cap_hit:
                            stopped = True
                            if result.attempts and result.attempts[-1].error == CAP_MESSAGE:
                                result.attempts.pop()  # the refused attempt was never sent
                            if generator.calls == calls_before:
                                break  # nothing new was paid for in this case: leave no row
                        best = result.best
                        row.update({
                            "score": "" if best is None or best.score is None else round(best.score, 6),
                            "pass": int(result.passed),
                            "attempts": len(result.attempts),
                            "api_calls": generator.calls - calls_before,
                            "cache_hits": generator.cache_hits - hits_before,
                            "detection_failures": result.detection_failures,
                            "generation_failures": result.generation_failures,
                            "seconds": round(time.monotonic() - started, 1),
                            "note": "; ".join(sorted({a.error for a in result.attempts if a.error})),
                        })
                        if best is not None and best.image is not None:
                            name = f"{face.name.replace('/', '__').rsplit('.', 1)[0]}__{style.id}__{mode}.png"
                            (outputs_dir / name).write_bytes(best.image)
                            row["output"] = f"outputs/{engine_name}/{name}"
                    writer.writerow(row)
                    file.flush()
                    print(f"{face.name} {style.id} {mode}: score={row.get('score', '')} pass={row['pass']} "
                          f"attempts={row['attempts']} api_calls={row.get('api_calls', 0)}")
                    if stopped:
                        break
                if stopped:
                    break
            if stopped:
                break

    print(f"\nEngine calls this run: {generator.calls}, cached results reused: {generator.cache_hits}.")
    if stopped:
        print(f"Stopped: the cap of {args.max_calls} paid API calls was reached. Finished rows are saved.")

    from report import main as write_report

    write_report()


if __name__ == "__main__":
    main()
