---
title: Rosilex Makeup
emoji: 💄
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: AI makeup try on that keeps the face the same person
---

# Rosilex makeup service

An API that applies one product or makeup style to a photo of a face and then measures how well
the result still matches the original person. It powers the AI testing page of the Rosilex
Shopify theme (`sections/makeup-demo.liquid`).

## Approach

1. **Upload checks.** JPEG, PNG or WebP up to 8 MB, checked from the file bytes. The photo must
   contain exactly one face (InsightFace). Uploads stay in memory and are never written to disk.
2. **What to apply.** Either a store product (title, brand, type, shade, description) or one of five
   preset styles. Product words decide where it goes: lips, eyes, cheeks, or facial skin for
   skincare.
3. **Masks.** MediaPipe face mesh landmarks give the lips, eyelids, cheeks and facial skin. Brows,
   eyeballs and the mouth opening are always protected, and so are the lips or eyes when the
   product is not for them.
4. **Generation.** OpenAI image edit (`gpt-image-1.5` by default, with `input_fidelity="high"`)
   receives the photo, the edit mask and a prompt that allows only that product and lists what must
   stay unchanged. Gemini 2.5 Flash Image can be used instead (`GENERATOR=gemini`); it takes no
   mask.
5. **Composite back.** The image model treats the mask as guidance only, so the result is combined
   with the original photo: makeup areas take the generated pixels inside the original feature
   outline; facial skin takes the generated skin (glow, sheen, finish) except wherever it is
   clearly darker than the original, where the original is kept, because skincare never adds dark
   features and a darker spot means the model moved a brow, a lip edge or a shadow; everything else
   is the original pixels. This is meant to stop doubled brows or larger lips.
6. **Identity check and retry.** The composited image is scored against the original (see below).
   Below the threshold, or with no face found, it is generated again, up to `MAX_ATTEMPTS` (at most
   3), and the best attempt is returned with its score and the number of attempts.

## How identity is measured

InsightFace `buffalo_l` (ArcFace `w600k_r50`, CPU) detects the face in the original and in the
result and computes a 512 value embedding of each. The identity score is the cosine similarity of
the two embeddings: 1.0 is identical, values near 0 are unrelated people.

## How the threshold was chosen

`eval/calibrate.py` compares every pair of photos in `eval/faces/<person>/`. Pairs of the same
person show how much a real second photo changes the score; pairs of different people are the
baseline. The pass threshold is the 5th percentile of same person scores, never below the level
that more than 1% of different person pairs reach. The current value in `config/threshold.json`
is **0.4175**, from 32 photos of 12 people in a subset of Labeled Faces in the Wild: 29 same person
pairs and 467 different person pairs. No different person pair reached it; the highest scored
0.1756.

## Status: not deployed

The service is not deployed. Every try on calls a paid image model, so it runs only locally for
testing and demos. Before a public launch it needs paid hosting with about 2 GB of memory (Docker),
a spending limit on the OpenAI account, and a privacy notice for the visitor log.

## Run it

Locally with Docker:

```bash
cp .env.example .env              # then fill in OPENAI_API_KEY
docker build -t makeup-service .
docker run --env-file .env -p 7860:7860 makeup-service
```

Tests (no API calls; Gemini, OpenAI and InsightFace are mocked):

```bash
docker build --target dev -t makeup-dev .
docker run --rm -v "$PWD:/work" -w /work makeup-dev python -m pytest -q
```

Evaluation: `python eval/calibrate.py` (no API calls), then `python eval/run_eval.py`, which writes
`eval/results/results_<engine>.csv` and `eval/results/summary.md`. Options: `--generator`
(`local`, `openai`, `gemini`), `--modes single,retry`, `--styles`, `--per-person`, `--limit`,
`--only-failed`, and `--max-calls`, a hard cap on paid API calls that is required for paid engines.
Paid results are cached in `eval/results/cache/`, so nothing is paid for twice, and every call is
logged with its token usage in `eval/results/calls_<engine>.jsonl`. The local engine is free.

## API

* `GET /api/styles`: the preset styles.
* `POST /api/makeup` (multipart): `image`, and `product` (JSON) or `style`; optional
  `seconds_on_site`. Returns the result image (base64), `identity_score`, `passed`, `threshold`,
  `attempts` and `tries_left`.
* `GET /admin/visitors`: the visitor log, with `Authorization: Bearer <ADMIN_TOKEN>`.
* `GET /health`.

## Settings

| Variable | Meaning |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI key (keep it a secret) |
| `GENERATOR` | `openai` or `gemini` |
| `OPENAI_IMAGE_MODEL` | Default `gpt-image-1.5` |
| `ALLOWED_ORIGINS` | Store domains allowed to call the API, comma separated |
| `MAX_ATTEMPTS` | Generations per request, 1 to 3 |
| `DAILY_GENERATION_LIMIT` | Image model calls per IP in any 24 hours (default 3, 0 is off) |
| `QUOTA_EXEMPT_IPS` | IPs with no daily limit, comma separated |
| `VISITORS_FILE` | Path of the visitor log, for example `/data/visitors.json` |
| `ADMIN_TOKEN` | Secret for reading the visitor log |
| `TRUST_FORWARDED_FOR` | `true` behind a proxy such as Hugging Face or Render |

## Limits

* Generation is done by OpenAI (or Gemini), so results vary from one run to the next, and the
  service depends on that provider being available and paid for.
* The threshold comes from a small test set of 32 photos of public figures. A larger set of
  consented photos that look like real customer selfies would give a more reliable value.
* The evaluation with the image model is a small sample: 10 photos, one style (soft glam), 14 paid
  calls. In it, 5 of 10 passed with one attempt and 6 of 10 with retries; one request was refused
  by OpenAI (HTTP 400). See `eval/results/summary.md`. This is not a reliable pass rate for real
  customers. The free local baseline engine (it paints on the photo and is not generative) passed
  160 of 160, which shows the scores an edit gets when it cannot change face shape.
* The test faces are public figures from Labeled Faces in the Wild (LFW): news photos at low
  resolution (250 x 250 pixels), 12 people, not consented customer selfies. They vary in angle,
  lighting and skin tone, but not in a controlled way.
* MediaPipe or InsightFace can miss faces in dark, tilted or partly covered photos; the API then
  asks for a clearer photo.
* Counts for the daily limit are kept in memory and start again when the service restarts. On a
  free Hugging Face Space the visitor log is on temporary disk and also starts again on restart.

## Privacy

* Photos are processed in memory for one request and never stored or logged. They are sent to the
  image provider (OpenAI or Google) for that request.
* The visitor log stores, per IP address: approximate city and country (from the offline DB-IP
  City Lite database, so IPs are not sent to a lookup service), first and last visit, the number of
  uses, and for each use the time, product, outcome and seconds on site. Tell visitors about it in
  the store's privacy policy.
* The daily limit keeps IPs only as salted hashes in memory.

IP geolocation by [DB-IP](https://db-ip.com), licensed under CC BY 4.0.
