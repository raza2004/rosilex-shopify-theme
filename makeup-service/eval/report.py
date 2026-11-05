"""Builds eval/results/summary.md, histogram.png and the worst case images from the evaluation files.

Reads results_<engine>.csv, calls_<engine>.jsonl (API usage), calibration.csv and config/threshold.json.

    python eval/report.py
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402

from common import FACES_DIR, RESULTS_DIR, SERVICE_ROOT  # noqa: E402

from app.styles import STYLES  # noqa: E402

PANEL_HEIGHT = 360
ENGINE_TITLES = {
    "local": "Local baseline engine (paints on the photo, not generative)",
    "openai": "OpenAI gpt-image-1.5 (generative, small sample)",
    "gemini": "Gemini 2.5 Flash Image (generative)",
}
# USD per 1M tokens for gpt-image-1.5, from https://developers.openai.com/api/docs/pricing (read 2026-11-05).
OPENAI_PRICES = {"text_in": 5.00, "image_in": 8.00, "text_out": 10.00, "image_out": 32.00}


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def score(row: dict) -> float | None:
    return float(row["score"]) if row.get("score") not in (None, "") else None


def describe(values: list[float]) -> str:
    if not values:
        return "n/a | n/a | n/a | n/a"
    return f"{statistics.mean(values):.3f} | {statistics.median(values):.3f} | {min(values):.3f} | {max(values):.3f}"


def rate(rows: list[dict]) -> str:
    if not rows:
        return "n/a"
    passed = sum(int(r["pass"]) for r in rows)
    return f"{passed / len(rows):.0%} ({passed}/{len(rows)})"


def reason(row: dict, threshold: float) -> str:
    note = row.get("note") or ""
    value = score(row)
    if note.startswith("input skipped"):
        return "mask or face detection failed on the input photo (" + note.split(":", 1)[-1].strip() + ")"
    if int(row.get("generation_failures") or 0) and value is None:
        return "the engine returned no image (" + note + ")"
    if value is None:
        return "no face detected in the result"
    if value < threshold:
        return "face drifted: identity score below the threshold"
    return "passed; lowest scores of the set"


def final_rows(rows: list[dict]) -> list[dict]:
    """Per photo and style, the retry result if there is one, otherwise the single attempt."""
    best: dict[tuple, dict] = {}
    for row in rows:
        key = (row["image"], row["style"])
        if key not in best or row["mode"] == "retry":
            best[key] = row
    return list(best.values())


def usage_totals(path: Path) -> dict:
    totals = {"calls": 0, "errors": 0, "with_usage": 0, "text_in": 0, "image_in": 0, "text_out": 0, "image_out": 0}
    if not path.exists():
        return totals
    for line in path.read_text(encoding="utf-8").splitlines():
        entry = json.loads(line)
        totals["calls"] += 1
        totals["errors"] += "error" in entry
        usage = entry.get("usage")
        if not usage:
            continue
        totals["with_usage"] += 1
        details_in = usage.get("input_tokens_details") or {}
        details_out = usage.get("output_tokens_details") or {}
        totals["text_in"] += details_in.get("text_tokens", 0)
        totals["image_in"] += details_in.get("image_tokens", usage.get("input_tokens", 0) - details_in.get("text_tokens", 0))
        totals["text_out"] += details_out.get("text_tokens", 0)
        totals["image_out"] += details_out.get("image_tokens", usage.get("output_tokens", 0) - details_out.get("text_tokens", 0))
    return totals


def side_by_side(original: Path, result: Path | None, out: Path) -> None:
    panels = [ImageOps.exif_transpose(Image.open(original)).convert("RGB")]
    if result and result.exists():
        panels.append(Image.open(result).convert("RGB"))
    panels = [p.resize((round(p.width * PANEL_HEIGHT / p.height), PANEL_HEIGHT)) for p in panels]
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 12 * (len(panels) - 1), PANEL_HEIGHT), "white")
    x = 0
    for panel in panels:
        canvas.paste(panel, (x, 0))
        x += panel.width + 12
    canvas.save(out, quality=88)


def histogram(engines: dict, calibration: list[dict], threshold: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=120)
    bins = [i / 50 for i in range(-10, 51)]
    series = [
        ([float(r["similarity"]) for r in calibration if r["same_person"] == "0"], "Different people (baseline)", "#9a98ad"),
        ([float(r["similarity"]) for r in calibration if r["same_person"] == "1"], "Same person, different photo", "#9FA9F2"),
    ]
    colors = {"local": "#8C6CF2", "openai": "#19153D", "gemini": "#5b4bb7"}
    for engine, rows in engines.items():
        values = [s for r in final_rows(rows) if (s := score(r)) is not None]
        series.append((values, f"{engine}: makeup result vs original", colors.get(engine, "#444")))
    for values, label, color in series:
        if values:
            ax.hist(values, bins=bins, alpha=0.6, label=f"{label} (n={len(values)})", color=color, density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
    ax.text(threshold, ax.get_ylim()[1] * 0.95, f" threshold {threshold:.4f}", va="top")
    ax.set_xlabel("Cosine similarity to the original face (InsightFace buffalo_l)")
    ax.set_ylabel("Density")
    ax.set_xlim(-0.2, 1.0)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def engine_section(engine: str, rows: list[dict], threshold: float) -> list[str]:
    ran = [r for r in rows if not (r.get("note") or "").startswith("input skipped")]
    single = [r for r in ran if r["mode"] == "single"]
    retry = [r for r in ran if r["mode"] == "retry"]
    final = final_rows(ran)
    lines = [f"## {ENGINE_TITLES.get(engine, engine)}", ""]
    if engine == "local":
        lines += [
            "This engine is a baseline, not generative AI: it paints the style onto the photo along face",
            "landmarks and cannot change face shape. It shows how high identity scores are for an edit that",
            "keeps the face by construction, for comparison with the generative engine.",
            "",
        ]
    else:
        lines += [
            f"**This is a small sample** ({len({r['image'] for r in ran})} photos, styles: "
            f"{', '.join(sorted({r['style'] for r in ran}))}). The numbers show how this sample behaved and are",
            "not a reliable estimate of the pass rate on real customers.",
            "",
        ]
    lines += [
        "| Style | Runs | Mean | Median | Min | Max | Pass rate, one attempt | Pass rate, with retries | Engine calls |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for style in STYLES:
        style_rows = [r for r in final if r["style"] == style.id]
        if not style_rows:
            continue
        values = [s for r in style_rows if (s := score(r)) is not None]
        calls = sum(int(r.get("api_calls") or r.get("attempts") or 0) for r in ran if r["style"] == style.id)
        lines.append(
            f"| {style.name} | {len(style_rows)} | {describe(values)} | "
            f"{rate([r for r in single if r['style'] == style.id])} | {rate(style_rows)} | {calls} |"
        )
    values = [s for r in final if (s := score(r)) is not None]
    calls = sum(int(r.get("api_calls") or r.get("attempts") or 0) for r in ran)
    lines += [
        f"| **All** | {len(final)} | {describe(values)} | **{rate(single)}** | **{rate(final)}** | {calls} |",
        "",
        "Scores and the pass rate with retries use the final result per photo and style: the best of the",
        f"retry run where one was made, otherwise the single attempt. Retry runs made: {len(retry)}.",
        "",
    ]
    return lines


def main() -> None:
    config = json.loads((SERVICE_ROOT / "config" / "threshold.json").read_text(encoding="utf-8"))
    threshold = float(config["threshold"])
    calibration = read_csv(RESULTS_DIR / "calibration.csv")
    engines = {p.stem.removeprefix("results_"): read_csv(p) for p in sorted(RESULTS_DIR.glob("results_*.csv"))}
    engines = {e: rows for e, rows in engines.items() if rows}
    if not engines:
        raise SystemExit("No results yet. Run eval/run_eval.py first.")
    histogram(engines, calibration, threshold, RESULTS_DIR / "histogram.png")

    genuine = [float(r["similarity"]) for r in calibration if r["same_person"] == "1"]
    impostor = [float(r["similarity"]) for r in calibration if r["same_person"] == "0"]
    all_rows = [r for rows in engines.values() for r in rows]
    lines = [
        "# Identity preservation results",
        "",
        "Test faces: a subset of Labeled Faces in the Wild (LFW), news photos of public figures at low",
        f"resolution (250 x 250 pixels), {len({r['person'] for r in all_rows})} people. They vary in angle,",
        "lighting and skin tone, but not in a controlled way. They are not customer selfies.",
        "",
        f"Pass threshold: **{threshold:.4f}** cosine similarity (InsightFace buffalo_l), chosen by",
        f"`eval/calibrate.py` as the {config.get('method', '')}.",
        "",
        "## Baseline: unedited photos",
        "",
        "| Pairs | Count | Mean | Median | Min | Max |",
        "| --- | --- | --- | --- | --- | --- |",
        f"| Different people | {len(impostor)} | {describe(impostor)} |",
        f"| Same person, two different photos | {len(genuine)} | {describe(genuine)} |",
        "",
        "![Similarity histogram](histogram.png)",
        "",
    ]
    for engine, rows in engines.items():
        lines += engine_section(engine, rows, threshold)
        if engine in ("openai", "gemini"):
            totals = usage_totals(RESULTS_DIR / f"calls_{engine}.jsonl")
            lines += [f"### API calls and cost ({engine})", "", f"* API calls made: {totals['calls']} (failed: {totals['errors']})"]
            if engine == "openai" and totals["with_usage"]:
                cost = sum(totals[k] * OPENAI_PRICES[k] for k in OPENAI_PRICES) / 1_000_000
                images = totals["with_usage"]
                lines += [
                    f"* Tokens reported by the API: {totals['text_in']} text input, {totals['image_in']} image input, "
                    f"{totals['text_out']} text output, {totals['image_out']} image output",
                    f"* Cost from those tokens at gpt-image-1.5 list prices: **${cost:.4f}** in total, "
                    f"**${cost / images:.4f}** per generated image ({images} images)",
                ]
            else:
                lines.append("* The API did not report token usage for these calls, so no cost is calculated here.")
            lines.append("")

        worst_dir = RESULTS_DIR / "worst" / engine
        worst_dir.mkdir(parents=True, exist_ok=True)
        ran = [r for r in rows if not (r.get("note") or "").startswith("input skipped")]
        worst = sorted(final_rows(ran), key=lambda r: (score(r) is not None, score(r) or 0))[:5]
        lines += [f"### Worst 5 results ({engine})", "", "Original on the left, result on the right.", ""]
        for index, row in enumerate(worst, 1):
            out = worst_dir / f"worst_{index}.jpg"
            side_by_side(FACES_DIR / row["image"], RESULTS_DIR / row["output"] if row.get("output") else None, out)
            value = score(row)
            shown = "no score" if value is None else f"{value:.3f}"
            lines += [
                f"{index}. `{row['image']}`, {row['style']}, {row['mode']}: **{shown}**, {reason(row, threshold)}.",
                "",
                f"   ![worst {index}](worst/{engine}/{out.name})",
                "",
            ]

    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
