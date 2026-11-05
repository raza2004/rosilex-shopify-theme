"""Chooses the identity pass threshold from data.

Compares every pair of photos in eval/faces/: pairs of the same person (genuine) and pairs of
different people (impostor).

The question for makeup is whether the edit changed the face more than ordinary variation between
two photos of the same person does. So the threshold is the --percentile (default 5th) of genuine
pair similarities: a passing result is at least as close to the original as 95% of real second
photos of the same person are. It is never set below the false accept threshold, the lowest value
that at most --far (default 1%) of different person pairs reach, so a pass always means the
result does not look like a different person to the recognition model.

Writes config/threshold.json (read by the API) and eval/results/calibration.csv.

    python eval/calibrate.py [--percentile 5] [--far 0.01]
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from datetime import datetime, timezone

import numpy as np

from common import RESULTS_DIR, SERVICE_ROOT, embed_faces, list_faces, make_embedder

from app.identity import cosine_similarity


def far_threshold(impostor: np.ndarray, far: float) -> float:
    """Smallest threshold t with mean(impostor >= t) <= far.

    At most k = floor(far * n) impostor pairs may pass, so t sits just above the (k+1)th highest score.
    """
    ranked = np.sort(impostor)[::-1]
    k = int(np.floor(far * ranked.size))
    if k >= ranked.size:
        return float(ranked[-1])
    return float(np.nextafter(ranked[k], np.inf))


def equal_error_rate(genuine: np.ndarray, impostor: np.ndarray) -> tuple[float, float]:
    """Threshold where false rejects of genuine pairs equal false accepts of impostor pairs."""
    best = (1.0, 0.0)
    gap = float("inf")
    for t in np.unique(np.concatenate([genuine, impostor])):
        frr, far = np.mean(genuine < t), np.mean(impostor >= t)
        if abs(frr - far) < gap:
            gap, best = abs(frr - far), ((frr + far) / 2, float(t))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--percentile", type=float, default=5, help="percentile of same person pair similarities to use")
    parser.add_argument("--far", type=float, default=0.01, help="allowed false accept rate on different person pairs (floor)")
    args = parser.parse_args()

    faces = embed_faces(list_faces(), make_embedder())
    usable = [f for f in faces if f.embedding is not None]
    for face in faces:
        if face.problem:
            print(f"skipped {face.name}: {face.problem}")

    rows, genuine, impostor = [], [], []
    for a, b in itertools.combinations(usable, 2):
        score = cosine_similarity(a.embedding, b.embedding)
        same = a.person == b.person
        (genuine if same else impostor).append(score)
        rows.append({"a": a.name, "b": b.name, "same_person": int(same), "similarity": round(score, 6)})

    genuine, impostor = np.array(genuine), np.array(impostor)
    if genuine.size == 0 or impostor.size == 0:
        raise SystemExit("Need at least two people, and at least one person with two or more photos.")

    far_floor = far_threshold(impostor, args.far)
    genuine_percentile = float(np.percentile(genuine, args.percentile))
    threshold = max(genuine_percentile, far_floor)
    eer, eer_threshold = equal_error_rate(genuine, impostor)
    rule = (
        f"{args.percentile:g}th percentile of same person pair similarities"
        if genuine_percentile >= far_floor
        else f"false accept floor ({args.far:.1%} of different person pairs), above the {args.percentile:g}th percentile of same person pairs"
    )
    stats = {
        "threshold": round(threshold, 4),
        "method": rule,
        "candidates": {
            f"genuine_p{args.percentile:g}": round(genuine_percentile, 4),
            f"far_{args.far:g}": round(far_floor, 4),
            "genuine_median": round(float(np.median(genuine)), 4),
            "eer_threshold": round(eer_threshold, 4),
        },
        "far_target": args.far,
        "far_measured": round(float(np.mean(impostor >= threshold)), 4),
        "same_person_pass_rate": round(float(np.mean(genuine >= threshold)), 4),
        "eer": round(eer, 4),
        "eer_threshold": round(eer_threshold, 4),
        "people": len({f.person for f in usable}),
        "photos": len(usable),
        "skipped_photos": len(faces) - len(usable),
        "genuine_pairs": int(genuine.size),
        "impostor_pairs": int(impostor.size),
        "genuine": {"mean": round(float(genuine.mean()), 4), "min": round(float(genuine.min()), 4), "max": round(float(genuine.max()), 4)},
        "impostor": {"mean": round(float(impostor.mean()), 4), "min": round(float(impostor.min()), 4), "max": round(float(impostor.max()), 4)},
        "model": "insightface buffalo_l (w600k_r50), cosine similarity of normed embeddings",
        "calibrated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "calibration.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["a", "b", "same_person", "similarity"])
        writer.writeheader()
        writer.writerows(rows)
    config = SERVICE_ROOT / "config" / "threshold.json"
    config.parent.mkdir(exist_ok=True)
    config.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(stats, indent=2))
    print(f"\nSuggested threshold {stats['threshold']} written to {config}")


if __name__ == "__main__":
    main()
