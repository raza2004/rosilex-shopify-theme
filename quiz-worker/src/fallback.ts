import type { Answers, Language, Product, Recommendation } from './types';

/**
 * Rule-based recommendation used when Gemini errors, times out or returns
 * something that fails validation. Scores each product by how many of its
 * tags (and, more weakly, title/description words) match the quiz answers.
 */

const normalise = (value: string) => value.toLowerCase().replace(/[^a-z0-9åäöæø]+/g, ' ').trim();

/** Each answer can match several tag spellings (e.g. "redness" also matches "calming"). */
const SYNONYMS: Record<string, string[]> = {
  dry: ['dry', 'dryness', 'hydrating', 'hydration', 'nourishing', 'barrier'],
  dryness: ['dry', 'dryness', 'hydrating', 'hydration', 'nourishing', 'barrier'],
  oily: ['oily', 'balancing', 'mattifying', 'clarifying', 'pores'],
  combination: ['combination', 'balancing'],
  normal: ['normal', 'all skin types'],
  sensitive: ['sensitive', 'gentle', 'calming', 'soothing', 'fragrance free'],
  redness: ['redness', 'calming', 'soothing', 'sensitive'],
  dullness: ['dullness', 'radiance', 'brightening', 'glow', 'vitamin c'],
  breakouts: ['breakouts', 'blemish', 'clarifying', 'pores', 'balancing'],
  blemishes: ['breakouts', 'blemish', 'clarifying', 'pores', 'balancing'],
  'fine lines': ['fine lines', 'firming', 'renewal', 'anti ageing', 'anti aging'],
  pores: ['pores', 'clarifying', 'refining'],
  'uneven tone': ['uneven tone', 'brightening', 'radiance'],
  cold: ['cold', 'winter', 'barrier', 'rich', 'nourishing'],
  humid: ['humid', 'lightweight', 'balancing'],
};

function termsFor(answer: string): string[] {
  const key = normalise(answer);
  return [key, ...(SYNONYMS[key] ?? [])];
}

function score(product: Product, answers: Answers): number {
  const tags = product.tags.map(normalise);
  const text = normalise(`${product.title} ${product.description}`);
  const has = (terms: string[], weight: number, textWeight: number) =>
    terms.some((t) => tags.includes(t)) ? weight : terms.some((t) => t.length > 3 && text.includes(t)) ? textWeight : 0;

  let total = 0;
  total += has(termsFor(answers.skinType), 3, 1);
  total += has(termsFor(answers.mainConcern), 4, 1);
  total += has(termsFor(answers.climate), 1, 0);
  if (/high|very|sensitive|yes/i.test(answers.sensitivity)) total += has(SYNONYMS.sensitive ?? [], 2, 1);
  if (tags.includes('all skin types') || tags.includes('bestseller')) total += 1;
  return total;
}

const COPY = {
  en: {
    reason: (title: string) => `${title} suits the skin type and concern you described.`,
    general: (title: string) => `${title} is a gentle, versatile step for most routines.`,
    tip: {
      short: 'Keep it simple: cleanse, then moisturise. Add sun protection every morning.',
      long: 'Cleanse, tone and apply serum on slightly damp skin, then seal with moisturiser. Finish mornings with sun protection.',
    },
  },
  sv: {
    reason: (title: string) => `${title} passar den hudtyp och det problem du beskrev.`,
    general: (title: string) => `${title} är ett milt och mångsidigt steg för de flesta rutiner.`,
    tip: {
      short: 'Håll det enkelt: rengör och återfukta. Lägg till solskydd varje morgon.',
      long: 'Rengör, tona och applicera serum på lätt fuktig hud, avsluta sedan med återfuktning. Avsluta morgonen med solskydd.',
    },
  },
} as const;

export function recommendByRules(products: Product[], answers: Answers, language: Language): Recommendation {
  const copy = COPY[language];
  const ranked = products
    .map((product, index) => ({ product, index, points: score(product, answers) }))
    // Highest score first; ties keep the merchant's product order.
    .sort((a, b) => b.points - a.points || a.index - b.index)
    .slice(0, 3);

  const short = /short|minimal|quick|kort|enkel/i.test(answers.routineLength);

  return {
    observations: [],
    picks: ranked.map(({ product, points }) => ({
      handle: product.handle,
      reason: points > 0 ? copy.reason(product.title) : copy.general(product.title),
    })),
    routineTip: short ? copy.tip.short : copy.tip.long,
  };
}
