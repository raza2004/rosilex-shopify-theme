export interface Env {
  GEMINI_API_KEY: string;
  GEMINI_MODEL?: string;
  /** Model for the one retry when the main model is overloaded. */
  GEMINI_FALLBACK_MODEL?: string;
  /** Comma-separated storefront origins allowed by CORS. */
  ALLOWED_ORIGIN: string;
  RATE_LIMIT: KVNamespace;
}

export type Language = 'sv' | 'en';

export interface Answers {
  skinType: string;
  mainConcern: string;
  sensitivity: string;
  routineLength: string;
  climate: string;
}

export interface Product {
  handle: string;
  title: string;
  description: string;
  tags: string[];
}

export interface ImageInput {
  mimeType: 'image/jpeg' | 'image/png';
  /** Base64 without any data: URL prefix. */
  base64: string;
}

export interface RecommendRequest {
  answers: Answers;
  products: Product[];
  language: Language;
  image: ImageInput | null;
}

export interface Pick {
  handle: string;
  reason: string;
}

export interface Recommendation {
  observations: string[];
  picks: Pick[];
  routineTip: string;
}

/** Thrown for bad input; becomes a 4xx response with a safe message. */
export class RequestError extends Error {
  constructor(
    message: string,
    readonly status = 400,
  ) {
    super(message);
  }
}
