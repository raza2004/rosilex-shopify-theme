"""What to apply: a preset makeup style or a product from the store, turned into a Look.

A Look is the prompt for the image model plus the face regions it may change. The regions
become the edit mask and the compositing mask, so a lipstick can only ever change the lips.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Look:
    id: str
    name: str
    prompt: str
    regions: frozenset[str]


@dataclass(frozen=True)
class Style:
    id: str
    name: str
    look: str
    regions: frozenset[str]


ALL_REGIONS = frozenset({"lips", "eyes", "cheeks", "skin"})

STYLES: tuple[Style, ...] = (
    Style(
        "natural",
        "Natural",
        "a natural everyday look: sheer skin tint that lets the real skin show through, "
        "a soft neutral blush, lightly groomed brows, a thin coat of brown mascara and a tinted lip balm",
        ALL_REGIONS,
    ),
    Style(
        "soft_glam",
        "Soft glam",
        "soft glam: satin finish foundation, warm bronzer and peach blush, shimmering champagne "
        "eyeshadow, defined lashes, softly sculpted brows and a glossy rosy nude lip",
        ALL_REGIONS,
    ),
    Style(
        "bold_lip",
        "Bold lip",
        "a bold lip: a precise, saturated classic red lipstick with clean edges",
        frozenset({"lips"}),
    ),
    Style(
        "smoky_eye",
        "Smoky eye",
        "a smoky eye: charcoal and soft black eyeshadow blended outward from the lash line, "
        "smudged liner, full lashes and a muted nude lip",
        frozenset({"eyes", "lips"}),
    ),
    Style(
        "bridal",
        "Bridal",
        "a classic bridal look: luminous long wear complexion, soft pink blush, "
        "delicate rose gold eyeshadow, fluttery lashes, defined brows and a satin pink lip",
        ALL_REGIONS,
    ),
)

STYLES_BY_ID = {style.id: style for style in STYLES}

KEEP_IDENTITY = (
    "The result must still be exactly the same person. Keep these unchanged: face shape, jawline, "
    "nose, eyes, eye shape and eye colour, eyebrow shape, lips shape, ears, skin texture including "
    "pores, freckles, moles and marks, facial expression, head pose and camera angle, hair, "
    "background, clothing, accessories, lighting and image framing.\n"
    "Do not move, redraw, enlarge or duplicate the eyebrows, eyes, nose or lips. "
    "Do not slim, smooth, reshape, de-age or beautify the face. Do not add or remove anything other "
    "than what is described. Return one photorealistic image at the same framing as the input."
)

PROMPT_TEMPLATE = "Edit this photo by applying makeup only. Apply {look}.\n" + KEEP_IDENTITY

REGION_WORDS = {
    "lips": "the lips",
    "eyes": "the eyelids and lash line (not the eyeballs)",
    "cheeks": "the cheeks",
    "skin": "the facial skin",
}


def prompt_for(style: Style) -> str:
    return PROMPT_TEMPLATE.format(look=style.look)


def look_for_style(style: Style) -> Look:
    return Look(style.id, style.name, prompt_for(style), style.regions)


# Product types and words that say where a product goes. Checked in order; first match wins per region.
PRODUCT_REGIONS = (
    ("lips", ("lip", "lipstick", "gloss", "lippen", "läpp")),
    ("eyes", ("eye", "shadow", "liner", "kohl", "mascara", "lash", "brow")),
    ("cheeks", ("blush", "bronzer", "highlighter", "contour", "cheek", "rouge")),
    ("skin", ("foundation", "concealer", "tint", "powder", "primer", "bb cream", "cc cream", "serum", "moistur",
              "cream", "spf", "sunscreen", "cleanser", "toner", "mask", "oil", "essence", "balm", "skin")),
)


@dataclass(frozen=True)
class Product:
    title: str
    vendor: str = ""
    type: str = ""
    description: str = ""
    shade: str = ""
    tags: tuple[str, ...] = ()


def _clean(value: object, limit: int) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))  # strip HTML
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()[:limit]


def parse_product(data: dict) -> Product:
    """Product details sent by the theme, trimmed to safe lengths. Raises ValueError without a title."""
    if not isinstance(data, dict):
        raise ValueError("product must be an object")
    title = _clean(data.get("title"), 150)
    if not title:
        raise ValueError("product title is required")
    tags = data.get("tags") or []
    return Product(
        title=title,
        vendor=_clean(data.get("vendor"), 80),
        type=_clean(data.get("type"), 80),
        description=_clean(data.get("description"), 600),
        shade=_clean(data.get("shade"), 80),
        tags=tuple(_clean(tag, 40) for tag in tags[:15] if _clean(tag, 40)) if isinstance(tags, list) else (),
    )


def regions_for_product(product: Product) -> frozenset[str]:
    """Where the product is applied, from its type, title and tags. Lip balm goes on the lips."""
    words = " ".join([product.type, product.title, *product.tags]).lower()
    found = {region for region, keys in PRODUCT_REGIONS if any(key in words for key in keys)}
    if "lips" in found and "balm" in words:
        found.discard("skin")
    return frozenset(found or {"skin"})


# What each kind of skincare product visibly does to skin, so different products give different results.
# Checked against the product type, title and tags; every match is used (a set combines them).
SKINCARE_EFFECTS = (
    (("cleanser", "cleansing", "face wash", "wash"),
     "freshly cleansed skin: clean, refreshed and slightly brighter, with less surface oil and a soft matte finish"),
    (("toner", "essence", "mist"),
     "refreshed, balanced skin: a slightly more even tone, softly calmed redness and a light, fresh dewiness"),
    (("serum", "ampoule", "booster"),
     "a visible healthy glow: more luminous, plumper and well hydrated looking skin with a brighter, radiant complexion"),
    (("moistur", "cream", "lotion", "balm", "gel"),
     "soft, supple, well moisturised skin with a smooth satin finish and a gentle healthy sheen; dry or flaky areas look calmer"),
    (("mask",),
     "the fresh after-mask look: deeply hydrated, dewy and radiant skin with calmer redness"),
    (("spf", "sunscreen", "sun "),
     "a light protective finish: an even, softly satin complexion, like a freshly applied lightweight sunscreen without a white cast"),
    (("oil",),
     "a luminous, glossy glow where the light catches the cheekbones, nose and forehead"),
    (("set", "ritual", "routine", "kit", "duo", "trio"),
     "the combined result of a full skincare routine: clean, balanced, well hydrated and glowing skin with a more even tone"),
)
DEFAULT_SKINCARE_EFFECT = "healthy, well cared for skin: fresh, hydrated and slightly more radiant"


def skincare_effect(product: Product) -> str:
    words = " ".join([product.type, product.title, *product.tags]).lower() + " "
    found = [effect for keys, effect in SKINCARE_EFFECTS if any(key in words for key in keys)]
    return "; ".join(found) or DEFAULT_SKINCARE_EFFECT


def look_for_product(product: Product) -> Look:
    regions = regions_for_product(product)
    areas = " and ".join(REGION_WORDS[r] for r in sorted(regions))
    details = [f"Product: {product.title}"]
    if product.vendor:
        details.append(f"Brand: {product.vendor}")
    if product.type:
        details.append(f"Type: {product.type}")
    if product.shade:
        details.append(f"Shade or variant: {product.shade}")
    if product.description:
        details.append(f"Description: {product.description}")
    effect = (
        f"Show the visible result of using this skincare product: {skincare_effect(product)}. "
        "Make the effect clearly visible but realistic, like a before and after photo taken minutes apart "
        "in the same light. Change only the skin's tone, glow and finish; keep pores, freckles and marks visible."
        if regions == {"skin"}
        else "Show how this exact product looks when worn, matching its colour, shade and finish."
    )
    prompt = (
        "Edit this photo to show the person wearing exactly one product, described below, and nothing else.\n"
        + "\n".join(details)
        + f"\nApply it only to {areas}. {effect}\n"
        + KEEP_IDENTITY
    )
    return Look("product", product.title, prompt, regions)
