const LIMIT = 10;
const WINDOW_SECONDS = 600;

interface Bucket {
  count: number;
  /** Unix seconds when the window started. */
  start: number;
}

/** Hashes the IP so the KV store never holds a raw address. */
async function keyFor(ip: string): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(`quiz-rl:${ip}`));
  return 'rl:' + [...new Uint8Array(digest)].slice(0, 16).map((b) => b.toString(16).padStart(2, '0')).join('');
}

/**
 * Fixed-window limit: 10 requests per 10 minutes per IP. KV is eventually
 * consistent, so a burst from one IP across Cloudflare locations can slip a
 * few extra requests through; that is acceptable for basic abuse protection.
 * Returns the seconds to wait when limited, or 0 when allowed.
 */
export async function checkRateLimit(kv: KVNamespace, ip: string): Promise<number> {
  const key = await keyFor(ip);
  const now = Math.floor(Date.now() / 1000);

  const stored = await kv.get<Bucket>(key, 'json');
  const bucket = stored && now - stored.start < WINDOW_SECONDS ? stored : { count: 0, start: now };

  if (bucket.count >= LIMIT) return bucket.start + WINDOW_SECONDS - now;

  bucket.count += 1;
  // KV's minimum TTL is 60 s; the window end is always at least that far away here.
  await kv.put(key, JSON.stringify(bucket), { expiration: Math.max(bucket.start + WINDOW_SECONDS, now + 60) });
  return 0;
}
