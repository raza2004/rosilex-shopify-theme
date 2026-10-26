// Exercises the quiz Worker with sample data. Start it first (npm run dev),
// then: npm test
//
// WORKER_URL defaults to the local dev server, ORIGIN to the store domain in
// wrangler.toml. Add --rate to also check the rate limit (it sends requests
// until the Worker answers 429, so run it last: it locks you out for 10 min).

const WORKER_URL = process.env.WORKER_URL ?? 'http://127.0.0.1:8787';
const ORIGIN = process.env.ORIGIN ?? 'https://rosexious.myshopify.com';
const CHECK_RATE_LIMIT = process.argv.includes('--rate');

const products = [
  { handle: 'rosilex-daily-cleanser', title: 'Rosilex Daily Cleanser', description: 'A gentle gel cleanser with birch sap.', tags: ['cleanser', 'all skin types', 'gentle'] },
  { handle: 'rosilex-daily-toner', title: 'Rosilex Daily Toner', description: 'A balancing toner with lingonberry fruit acids.', tags: ['toner', 'oily', 'combination', 'pores'] },
  { handle: 'rosilex-daily-serum', title: 'Rosilex Daily Serum', description: 'A brightening serum with sea buckthorn.', tags: ['serum', 'dullness', 'brightening', 'uneven tone'] },
  { handle: 'rosilex-daily-moisturiser', title: 'Rosilex Daily Moisturiser', description: 'A rich cream with cloudberry seed oil.', tags: ['moisturiser', 'dry', 'barrier', 'cold'] },
  { handle: 'rosilex-facial-mask', title: 'Rosilex Facial Mask', description: 'A calming weekly mask.', tags: ['mask', 'sensitive', 'redness', 'calming'] },
];

const answers = { skinType: 'dry', mainConcern: 'redness', sensitivity: 'high', routineLength: 'standard', climate: 'cold' };

// 1x1 white JPEG: a valid image that is "unclear", so the model should ignore it.
const TINY_JPEG = '/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAMCAgICAgMCAgIDAwMDBAYEBAQEBAgGBgUGCQgKCgkICQkKDA8MCgsOCwkJDRENDg8QEBEQCgwSExIQEw8QEBD/yQALCAABAAEBAREA/8wABgAQEAX/2gAIAQEAAD8A0s8g/9k=';

let failures = 0;
let skipped = 0;
const check = (name, condition, detail = '') => {
  console.log(`${condition ? 'PASS' : 'FAIL'}  ${name}${detail ? `  (${detail})` : ''}`);
  if (!condition) failures += 1;
};
/** Like check, but a 429 means an earlier run used up the limit: skip instead of fail. */
const checkResponse = (name, res, condition, detail = '') => {
  if (res.status === 429 && !name.startsWith('rate limit')) {
    skipped += 1;
    console.log(`SKIP  ${name}  (rate limited by an earlier run)`);
    return;
  }
  check(name, condition, detail);
};

const post = (body, origin = ORIGIN) =>
  fetch(`${WORKER_URL}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Origin: origin },
    body: JSON.stringify(body),
  });

// CORS
const preflight = await fetch(`${WORKER_URL}/recommend`, {
  method: 'OPTIONS',
  headers: { Origin: ORIGIN, 'Access-Control-Request-Method': 'POST' },
});
check('preflight from the store is allowed', preflight.status === 204 && preflight.headers.get('access-control-allow-origin') === ORIGIN);

const foreign = await post({ answers, products, language: 'en' }, 'https://evil.example');
check('other origins are rejected', foreign.status === 403, `HTTP ${foreign.status}`);

// Recommendations
for (const language of ['en', 'sv']) {
  const started = Date.now();
  const res = await post({ answers, products, language });
  const data = await res.json();
  const handles = new Set(products.map((p) => p.handle));
  checkResponse(
    `recommendation (${language})`,
    res,
    res.status === 200 && data.picks?.length === 3 && data.picks.every((p) => handles.has(p.handle)),
    `HTTP ${res.status}, source=${data.source}, ${((Date.now() - started) / 1000).toFixed(1)}s`,
  );
  if (res.status === 200) {
    console.log(`      picks: ${data.picks.map((p) => p.handle).join(', ')}`);
    console.log(`      reason: ${data.picks[0].reason}`);
    console.log(`      tip: ${data.routineTip}`);
  }
}

// Image rules
const noConsent = await post({ answers, products, language: 'en', image: TINY_JPEG });
checkResponse('image without consent is rejected', noConsent, noConsent.status === 403, `HTTP ${noConsent.status}`);

const notAnImage = await post({ answers, products, language: 'en', consent: true, image: btoa('this is not an image') });
checkResponse('non-JPEG/PNG data is rejected', notAnImage, notAnImage.status === 415, `HTTP ${notAnImage.status}`);

const huge = await post({ answers, products, language: 'en', consent: true, image: 'A'.repeat(2_100_000) });
checkResponse('images over 1.5 MB are rejected', huge, huge.status === 413, `HTTP ${huge.status}`);

const withImage = await post({ answers, products, language: 'en', consent: true, image: TINY_JPEG });
const withImageData = await withImage.json();
checkResponse('consented JPEG is accepted', withImage, withImage.status === 200 && withImageData.picks?.length === 3, `HTTP ${withImage.status}, source=${withImageData.source}`);

// Rate limit
if (CHECK_RATE_LIMIT) {
  let limited = null;
  for (let i = 0; i < 12 && !limited; i += 1) {
    const res = await post({ language: 'en' }); // invalid on purpose: still counts, no Gemini call
    if (res.status === 429) limited = res;
  }
  check('rate limit answers 429 with Retry-After', Boolean(limited?.headers.get('retry-after')), limited ? `Retry-After ${limited.headers.get('retry-after')}s` : 'never limited');
}

if (skipped) console.log(`\n${skipped} check(s) skipped: the rate limit resets 10 minutes after the first request (locally, restart "npm run dev" after deleting .wrangler/state to reset it now).`);
console.log(failures ? `\n${failures} check(s) failed` : '\nAll checks passed');
process.exit(failures ? 1 : 0);
