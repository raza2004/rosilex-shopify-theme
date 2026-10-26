import { RequestError, type Answers, type ImageInput, type Product, type RecommendRequest } from './types';

const MAX_IMAGE_BYTES = 1.5 * 1024 * 1024;
const MAX_PRODUCTS = 60;
const ANSWER_KEYS: (keyof Answers)[] = ['skinType', 'mainConcern', 'sensitivity', 'routineLength', 'climate'];

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === 'object' && value !== null && !Array.isArray(value);

/** Trims and caps a string field; free text is length-limited to keep prompts small. */
function text(value: unknown, field: string, max: number, required = true): string {
  if (typeof value !== 'string' || (required && !value.trim())) {
    throw new RequestError(`"${field}" must be a non-empty string`);
  }
  return value.trim().slice(0, max);
}

function parseAnswers(value: unknown): Answers {
  if (!isRecord(value)) throw new RequestError('"answers" must be an object');

  const answers = {} as Answers;
  for (const key of ANSWER_KEYS) {
    answers[key] = text(value[key], `answers.${key}`, 120);
  }
  return answers;
}

function parseProducts(value: unknown): Product[] {
  if (!Array.isArray(value) || value.length < 3) {
    throw new RequestError('"products" must be an array of at least 3 products');
  }
  if (value.length > MAX_PRODUCTS) {
    throw new RequestError(`"products" can hold at most ${MAX_PRODUCTS} products`);
  }

  const seen = new Set<string>();
  return value.map((item, index) => {
    if (!isRecord(item)) throw new RequestError(`products[${index}] must be an object`);

    const handle = text(item.handle, `products[${index}].handle`, 120);
    if (seen.has(handle)) throw new RequestError(`Duplicate product handle "${handle}"`);
    seen.add(handle);

    const tags = Array.isArray(item.tags)
      ? item.tags.filter((tag): tag is string => typeof tag === 'string').map((tag) => tag.trim().slice(0, 60))
      : [];

    return {
      handle,
      title: text(item.title, `products[${index}].title`, 200),
      description: text(item.description ?? '', `products[${index}].description`, 600, false),
      tags: tags.slice(0, 30),
    };
  });
}

/** Detects the real file type from its first bytes, not from what the client claims. */
function sniffImage(bytes: Uint8Array): ImageInput['mimeType'] | null {
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) return 'image/jpeg';
  if (bytes[0] === 0x89 && bytes[1] === 0x50 && bytes[2] === 0x4e && bytes[3] === 0x47) return 'image/png';
  return null;
}

function parseImage(value: unknown): ImageInput {
  if (typeof value !== 'string' || !value) throw new RequestError('"image" must be a base64 string');

  // Accept both raw base64 and data URLs (data:image/jpeg;base64,...).
  const base64 = value.replace(/^data:[^;,]+;base64,/, '').replace(/\s/g, '');

  // Reject oversized payloads before decoding: 4 base64 chars carry 3 bytes.
  if ((base64.length * 3) / 4 > MAX_IMAGE_BYTES + 3) {
    throw new RequestError('Image is larger than 1.5 MB', 413);
  }

  let bytes: Uint8Array;
  try {
    bytes = Uint8Array.from(atob(base64), (char) => char.charCodeAt(0));
  } catch {
    throw new RequestError('"image" is not valid base64');
  }

  if (bytes.byteLength > MAX_IMAGE_BYTES) throw new RequestError('Image is larger than 1.5 MB', 413);

  const mimeType = sniffImage(bytes);
  if (!mimeType) throw new RequestError('Image must be a JPEG or PNG', 415);

  return { mimeType, base64 };
}

export function parseRequest(body: unknown): RecommendRequest {
  if (!isRecord(body)) throw new RequestError('Request body must be a JSON object');

  const language = body.language === 'sv' ? 'sv' : body.language === 'en' ? 'en' : null;
  if (!language) throw new RequestError('"language" must be "sv" or "en"');

  const hasImage = body.image !== undefined && body.image !== null && body.image !== '';
  if (hasImage && body.consent !== true) {
    throw new RequestError('Consent is required to analyse a photo', 403);
  }

  return {
    answers: parseAnswers(body.answers),
    products: parseProducts(body.products),
    language,
    image: hasImage ? parseImage(body.image) : null,
  };
}
