# Rosilex quiz worker

A Cloudflare Worker that turns skincare quiz answers (and, optionally, a
photo the visitor consents to share) into three product picks from your
store, using Google Gemini. If Gemini is slow, overloaded or returns anything
invalid, it falls back to a rule-based pick from product tags, so the quiz
always gets an answer.

## Endpoint

`POST /recommend` with JSON:

```json
{
  "answers": {
    "skinType": "dry",
    "mainConcern": "redness",
    "sensitivity": "high",
    "routineLength": "standard",
    "climate": "cold"
  },
  "products": [
    { "handle": "rosilex-daily-cleanser", "title": "Daily Cleanser", "description": "...", "tags": ["cleanser", "gentle"] }
  ],
  "language": "sv",
  "image": "<optional base64 JPEG or PNG, max 1.5 MB>",
  "consent": true
}
```

Response:

```json
{
  "observations": ["..."],
  "picks": [{ "handle": "rosilex-daily-cleanser", "reason": "..." }],
  "routineTip": "...",
  "source": "ai"
}
```

`source` is `"ai"` when Gemini answered and `"rules"` when the fallback did.

| Status | Meaning |
| --- | --- |
| 400 | Invalid body (missing answers, fewer than 3 products, bad language) |
| 403 | Image sent without `consent: true`, or the request came from another site |
| 413 | Image over 1.5 MB, or request too large |
| 415 | Image is not a JPEG or PNG (checked from the file bytes) |
| 429 | More than 10 requests in 10 minutes from one visitor (`Retry-After` header says how long to wait) |

## Privacy

- The image and answers are sent to Gemini for that one request and never
  stored or logged. The Worker only logs short fixed messages such as
  `gemini: HTTP 503`.
- The rate limiter stores a SHA-256 hash of the visitor's IP, never the IP
  itself, and it expires after 10 minutes.
- Gemini is instructed to stay cosmetic: no diagnoses or medical conditions,
  no comments on age, ethnicity or looks, only products from your list, ignore
  unclear photos, and suggest a dermatologist neutrally for concerning symptoms.

## Setup

You need Node.js 20+ and a Cloudflare account.

```bash
cd quiz-worker
npm install
```

### 1. Run it locally

Put your Gemini key in `quiz-worker/.env` (already created, and git-ignored):

```
GEMINI_API_KEY=your-key-here
```

Then:

```bash
npm run dev          # starts the Worker at http://127.0.0.1:8787
npm test             # in a second terminal: runs the sample requests
npm test -- --rate   # also checks the rate limit (locks you out for 10 minutes)
```

### 2. Deploy to Cloudflare

```bash
npx wrangler login
npx wrangler kv namespace create RATE_LIMIT
```

Copy the `id` it prints into `wrangler.toml` (replace `REPLACE_WITH_KV_NAMESPACE_ID`).

Set your store domains in `wrangler.toml`, comma-separated, exactly as they
appear in the browser:

```toml
ALLOWED_ORIGIN = "https://rosexious.myshopify.com,https://www.your-domain.com"
```

Store the Gemini key as a Cloudflare secret (never in `wrangler.toml`):

```bash
npx wrangler secret put GEMINI_API_KEY
npm run deploy
```

Wrangler prints the Worker URL, for example
`https://rosilex-quiz.<your-subdomain>.workers.dev`. Test it with:

```bash
WORKER_URL=https://rosilex-quiz.<your-subdomain>.workers.dev npm test
```

### 3. Call it from the theme

```js
const response = await fetch('https://rosilex-quiz.<your-subdomain>.workers.dev/recommend', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ answers, products, language: Shopify.locale === 'sv' ? 'sv' : 'en' }),
});
const { picks, observations, routineTip } = await response.json();
```

## Models

| Setting | Default | Notes |
| --- | --- | --- |
| `GEMINI_MODEL` | `gemini-3.5-flash` | About 2 to 3 s per answer in testing |
| `GEMINI_FALLBACK_MODEL` | `gemini-3.1-flash-lite` | One retry when the main model is slow or overloaded |

The main model gets 12 s. If it times out or is overloaded (HTTP 429/5xx),
the fallback model gets 10 s, and only then do the tag-based rules answer.
`gemini-3.8-flash` is newer but took 40 to 65 s per answer in testing, too
slow for a quiz.

## Files

- `src/index.ts`: routing, CORS, rate limit, error handling
- `src/validate.ts`: request validation, consent, image size and type checks
- `src/gemini.ts`: prompt, safety instructions, response schema, output validation
- `src/fallback.ts`: rule-based picks from product tags
- `src/rate-limit.ts`: 10 requests per 10 minutes per visitor, using KV
- `test/sample-request.mjs`: sample data and checks
