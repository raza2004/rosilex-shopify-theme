import { recommendByRules } from './fallback';
import { recommendWithGemini } from './gemini';
import { checkRateLimit } from './rate-limit';
import { RequestError, type Env } from './types';
import { parseRequest } from './validate';

/** Base64 of a 1.5 MB image is ~2 MB; leave room for answers and products. */
const MAX_BODY_BYTES = 2.6 * 1024 * 1024;

function allowedOrigin(request: Request, env: Env): string | null {
  const origin = request.headers.get('Origin');
  if (!origin) return null;
  const allowed = env.ALLOWED_ORIGIN.split(',').map((o) => o.trim().replace(/\/$/, ''));
  return allowed.includes(origin) ? origin : null;
}

function json(body: unknown, status: number, origin: string | null, extra: HeadersInit = {}): Response {
  const headers = new Headers({
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
    Vary: 'Origin',
    ...extra,
  });
  if (origin) headers.set('Access-Control-Allow-Origin', origin);
  return new Response(JSON.stringify(body), { status, headers });
}

export default {
  async fetch(request, env): Promise<Response> {
    const url = new URL(request.url);
    const origin = allowedOrigin(request, env);

    if (url.pathname !== '/recommend') return json({ error: 'Not found' }, 404, origin);

    // Browsers only: any request carrying an Origin must come from the store.
    if (request.headers.get('Origin') && !origin) return json({ error: 'Origin not allowed' }, 403, null);

    if (request.method === 'OPTIONS') {
      return new Response(null, {
        status: origin ? 204 : 403,
        headers: origin
          ? {
              'Access-Control-Allow-Origin': origin,
              'Access-Control-Allow-Methods': 'POST, OPTIONS',
              'Access-Control-Allow-Headers': 'Content-Type',
              'Access-Control-Max-Age': '86400',
              Vary: 'Origin',
            }
          : { Vary: 'Origin' },
      });
    }

    if (request.method !== 'POST') return json({ error: 'Method not allowed' }, 405, origin, { Allow: 'POST, OPTIONS' });

    const ip = request.headers.get('CF-Connecting-IP') ?? 'unknown';
    const retryAfter = await checkRateLimit(env.RATE_LIMIT, ip);
    if (retryAfter > 0) {
      return json({ error: 'Too many requests. Please try again later.' }, 429, origin, { 'Retry-After': String(retryAfter) });
    }

    const declaredLength = Number(request.headers.get('Content-Length') ?? 0);
    if (declaredLength > MAX_BODY_BYTES) return json({ error: 'Request is too large' }, 413, origin);

    try {
      const raw = await request.text();
      if (raw.length > MAX_BODY_BYTES) throw new RequestError('Request is too large', 413);

      let body: unknown;
      try {
        body = JSON.parse(raw);
      } catch {
        throw new RequestError('Request body must be valid JSON');
      }

      const input = parseRequest(body);
      const ai = await recommendWithGemini(env, input);
      const result = ai ?? recommendByRules(input.products, input.answers, input.language);

      return json({ ...result, source: ai ? 'ai' : 'rules' }, 200, origin);
    } catch (error) {
      if (error instanceof RequestError) return json({ error: error.message }, error.status, origin);
      // No details: the error could reference request content.
      console.error('recommend: unexpected error');
      return json({ error: 'Something went wrong' }, 500, origin);
    }
  },
} satisfies ExportedHandler<Env>;
