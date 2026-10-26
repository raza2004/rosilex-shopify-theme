import type { Env, Product, Recommendation, RecommendRequest } from './types';

const DEFAULT_MODEL = 'gemini-3.5-flash';
/** Used for the single retry when the main model is overloaded (429/5xx). */
const DEFAULT_FALLBACK_MODEL = 'gemini-3.1-flash-lite';
/** Time limits per attempt; together they stay under the quiz's 30 s client timeout. */
const PRIMARY_TIMEOUT_MS = 12_000;
const FALLBACK_TIMEOUT_MS = 10_000;

const SYSTEM_INSTRUCTION = `You are a cosmetic skincare advisor for a Nordic skincare brand.

Rules you must always follow:
- Give cosmetic, non-medical guidance only. Never diagnose and never name or suggest medical or skin conditions or diseases.
- Never comment on the person's age, ethnicity, gender, attractiveness or any other personal trait.
- Recommend only products from the provided product list, referring to them by their exact handle.
- If a photo is provided but it is unclear, too dark, not a face, or contains more than one person, ignore it completely and rely only on the quiz answers. Do not mention the photo in that case.
- Observations describe only visible, cosmetic qualities of skin (for example shine, visible dryness, uneven-looking tone), in gentle neutral language. If there is no usable photo, base observations on the answers instead.
- If the answers mention a concerning symptom (for example pain, bleeding, a changing mole, a spreading rash, or sudden severe reactions), add one neutral sentence in routineTip suggesting the person see a dermatologist, without guessing what it is.
- Write all text in the requested language.
- Keep each observation under 20 words, each reason under 30 words and the routine tip under 45 words.`;

function buildPrompt({ answers, products, language }: RecommendRequest): string {
  const productList = products
    .map((p) => `- handle: ${p.handle}\n  title: ${p.title}\n  tags: ${p.tags.join(', ') || 'none'}\n  description: ${p.description || 'none'}`)
    .join('\n');

  return [
    `Language for all text: ${language === 'sv' ? 'Swedish' : 'English'}.`,
    '',
    'Quiz answers:',
    `- Skin type: ${answers.skinType}`,
    `- Main concern: ${answers.mainConcern}`,
    `- Sensitivity: ${answers.sensitivity}`,
    `- Preferred routine length: ${answers.routineLength}`,
    `- Climate: ${answers.climate}`,
    '',
    'Products (choose exactly 3 different handles from this list):',
    productList,
  ].join('\n');
}

function responseSchema(products: Product[]) {
  return {
    type: 'OBJECT',
    properties: {
      observations: { type: 'ARRAY', items: { type: 'STRING' }, maxItems: 3 },
      picks: {
        type: 'ARRAY',
        minItems: 3,
        maxItems: 3,
        items: {
          type: 'OBJECT',
          properties: {
            // The enum stops the model inventing products; we still validate below.
            handle: { type: 'STRING', enum: products.map((p) => p.handle) },
            reason: { type: 'STRING' },
          },
          required: ['handle', 'reason'],
        },
      },
      routineTip: { type: 'STRING' },
    },
    required: ['observations', 'picks', 'routineTip'],
  };
}

/** Checks the model output strictly; returns null if anything is off so the caller falls back. */
export function validateRecommendation(value: unknown, products: Product[]): Recommendation | null {
  if (typeof value !== 'object' || value === null) return null;
  const { observations, picks, routineTip } = value as Record<string, unknown>;

  if (!Array.isArray(observations) || !observations.every((o) => typeof o === 'string')) return null;
  if (typeof routineTip !== 'string' || !routineTip.trim()) return null;
  if (!Array.isArray(picks) || picks.length !== 3) return null;

  const handles = new Set(products.map((p) => p.handle));
  const chosen = new Set<string>();
  for (const pick of picks) {
    if (typeof pick !== 'object' || pick === null) return null;
    const { handle, reason } = pick as Record<string, unknown>;
    if (typeof handle !== 'string' || !handles.has(handle) || chosen.has(handle)) return null;
    if (typeof reason !== 'string' || !reason.trim()) return null;
    chosen.add(handle);
  }

  return {
    observations: (observations as string[]).slice(0, 3).map((o) => o.trim().slice(0, 240)),
    picks: (picks as { handle: string; reason: string }[]).map(({ handle, reason }) => ({
      handle,
      reason: reason.trim().slice(0, 320),
    })),
    routineTip: routineTip.trim().slice(0, 480),
  };
}

/**
 * Asks Gemini for a recommendation. Returns null on any failure (HTTP error,
 * timeout, blocked or malformed output) so the caller can use the rule-based
 * fallback. Never logs the request content.
 */
export async function recommendWithGemini(env: Env, request: RecommendRequest): Promise<Recommendation | null> {
  if (!env.GEMINI_API_KEY) {
    console.error('gemini: GEMINI_API_KEY is not configured');
    return null;
  }

  const parts: Record<string, unknown>[] = [{ text: buildPrompt(request) }];
  if (request.image) {
    parts.push({ inline_data: { mime_type: request.image.mimeType, data: request.image.base64 } });
  }

  const model = env.GEMINI_MODEL || DEFAULT_MODEL;
  const fallbackModel = env.GEMINI_FALLBACK_MODEL || DEFAULT_FALLBACK_MODEL;

  const body = JSON.stringify({
    systemInstruction: { parts: [{ text: SYSTEM_INSTRUCTION }] },
    contents: [{ role: 'user', parts }],
    generationConfig: {
      temperature: 0.4,
      responseMimeType: 'application/json',
      responseSchema: responseSchema(request.products),
      thinkingConfig: { thinkingLevel: 'low' },
    },
  });

  /**
   * One call to one model with its own time limit. Returns the response, or
   * the string 'retry' when the failure is worth retrying on the lighter
   * model (timeout, network error, 429 or 5xx).
   */
  const attempt = async (modelName: string, timeoutMs: number): Promise<Response | 'retry'> => {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(modelName)}:generateContent`,
        {
          method: 'POST',
          signal: controller.signal,
          headers: { 'Content-Type': 'application/json', 'x-goog-api-key': env.GEMINI_API_KEY },
          body,
        },
      );
      if (response.status === 429 || response.status >= 500) {
        console.error(`gemini: HTTP ${response.status} from ${modelName}`);
        return 'retry';
      }
      // Read the body inside the time limit too.
      return new Response(await response.text(), { status: response.status });
    } catch (error) {
      console.error(`gemini: ${modelName} ${error instanceof Error && error.name === 'AbortError' ? 'timed out' : 'request failed'}`);
      return 'retry';
    } finally {
      clearTimeout(timer);
    }
  };

  try {
    // The main model gets most of the budget; if it is slow or overloaded the
    // lighter model (usually 1-2 s) gets a chance before the rules take over.
    let response = await attempt(model, PRIMARY_TIMEOUT_MS);
    if (response === 'retry') response = await attempt(fallbackModel, FALLBACK_TIMEOUT_MS);
    if (response === 'retry') return null;

    if (!response.ok) {
      // Status only: the error body can echo parts of the request.
      console.error(`gemini: HTTP ${response.status}`);
      return null;
    }

    const data = (await response.json()) as {
      candidates?: { content?: { parts?: { text?: string }[] }; finishReason?: string }[];
    };
    const text = data.candidates?.[0]?.content?.parts?.find((p) => typeof p.text === 'string')?.text;
    if (!text) {
      console.error(`gemini: no text in response (finishReason ${data.candidates?.[0]?.finishReason ?? 'none'})`);
      return null;
    }

    const result = validateRecommendation(JSON.parse(text), request.products);
    if (!result) console.error('gemini: response failed validation');
    return result;
  } catch {
    console.error('gemini: unreadable response');
    return null;
  }
}
