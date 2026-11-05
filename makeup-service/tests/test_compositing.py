import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app import compositing
from app.regions import RegionMasks
from app.generator import EditRequest, OpenAIGenerator
from app.styles import Product, look_for_product, parse_product, regions_for_product


def photo(width: int, height: int) -> Image.Image:
    rng = np.random.default_rng(3)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


def png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.parametrize("size, expected", [((800, 800), (1024, 1024)), ((600, 900), (1024, 1536)), ((1200, 700), (1536, 1024))])
def test_model_size_follows_orientation(size, expected):
    assert compositing.model_size(*size) == expected


def masks(makeup=None, skin=None, protect=None, shape=(900, 600)):
    zero = np.zeros(shape, np.float32)
    return RegionMasks(zero if makeup is None else makeup, zero if skin is None else skin, zero if protect is None else protect)


def test_pixels_outside_the_mask_stay_original():
    original = photo(600, 900)
    region = np.zeros((900, 600), np.float32)
    region[400:500, 250:350] = 1  # a "lips" box
    prepared = compositing.prepare(original, masks(makeup=region))

    # The model returns something completely different everywhere.
    generated = Image.new("RGB", prepared.size, (255, 0, 255))
    result = np.asarray(compositing.composite(original, png(generated), prepared)).astype(int)
    base = np.asarray(original).astype(int)

    far = np.ones((900, 600), bool)
    far[340:560, 190:410] = False  # region plus its soft edge
    assert np.array_equal(result[far], base[far])
    centre = result[440:460, 290:310]
    assert np.allclose(centre, [255, 0, 255], atol=2)


def test_mask_is_transparent_only_where_editing_is_allowed():
    original = photo(800, 800)
    region = np.zeros((800, 800), np.float32)
    region[300:400, 300:500] = 1
    prepared = compositing.prepare(original, masks(makeup=region, shape=(800, 800)))
    alpha = np.asarray(Image.open(io.BytesIO(prepared.mask_png)))[..., 3]
    assert alpha.shape == (1024, 1024)
    assert alpha[450, 512] == 0  # inside the region (scaled by 1024/800)
    assert alpha[50, 50] == 255  # far outside
    assert 0.01 < (alpha == 0).mean() < 0.1


def test_portrait_photo_is_padded_not_stretched():
    prepared = compositing.prepare(photo(600, 1000), masks(shape=(1000, 600)))
    left, top, right, bottom = prepared.box
    assert prepared.size == (1024, 1536)
    assert (right - left) / (bottom - top) == pytest.approx(0.6, abs=0.01)


@pytest.mark.parametrize(
    "product, regions",
    [
        (Product("Velvet lipstick", type="Lipstick"), {"lips"}),
        (Product("Lip balm", type="Lip care"), {"lips"}),
        (Product("Night kohl", type="Eyeliner"), {"eyes"}),
        (Product("Peach blush"), {"cheeks"}),
        (Product("Rosilex Daily Serum", type="Serum"), {"skin"}),
        (Product("Mystery item"), {"skin"}),
    ],
)
def test_product_regions(product, regions):
    assert regions_for_product(product) == regions


def test_product_text_is_cleaned_and_trimmed():
    product = parse_product({"title": " <b>Rouge</b>\n", "description": "x" * 5000, "tags": ["lip", "<i>new</i>"]})
    assert product.title == "Rouge"
    assert len(product.description) == 600
    assert product.tags == ("lip", "new")
    assert "Rouge" in look_for_product(product).prompt


class FakeImages:
    def __init__(self):
        self.calls = []

    def edit(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(b64_json="aGVsbG8=")])


@pytest.mark.parametrize("model, fidelity", [("gpt-image-1.5", "high"), ("gpt-image-2.5-sunburst", None)])
def test_input_fidelity_only_for_models_that_support_it(model, fidelity):
    generator = OpenAIGenerator.__new__(OpenAIGenerator)
    generator._model = model
    generator._client = SimpleNamespace(images=FakeImages())
    result = generator.generate(EditRequest(b"img", b"mask", (1024, 1536), "prompt"))
    call = generator._client.images.calls[0]
    assert call.get("input_fidelity") == fidelity
    assert call["size"] == "1024x1536" and call["mask"][1] == b"mask"
    assert result.data == b"hello"


def face_like(brow_y: int, lip_half_height: int, skin: int = 150) -> Image.Image:
    """Flat skin with a dark brow bar and a red lip ellipse: enough structure to see doubling."""
    import cv2

    canvas = np.full((900, 600, 3), skin, np.uint8)
    canvas[brow_y:brow_y + 12, 180:420] = (60, 40, 30)
    cv2.ellipse(canvas, (300, 600), (70, lip_half_height), 0, 0, 360, (170, 60, 70), -1)
    return Image.fromarray(canvas)


def test_skin_product_cannot_double_brows_or_grow_lips():
    """The model redraws the face: brow 35 px higher (outside the protected zone), lips fuller, brighter skin."""
    import cv2

    original = face_like(brow_y=250, lip_half_height=22)
    generated = face_like(brow_y=215, lip_half_height=34, skin=175)
    skin = np.zeros((900, 600), np.float32)
    skin[100:800, 100:500] = 1
    protect = np.zeros((900, 600), np.float32)
    protect[240:275, 170:430] = 1  # original brow with a small margin
    protect[560:640, 210:390] = 1  # lips with margin
    protect = np.clip(cv2.GaussianBlur(protect, (0, 0), 4), 0, 1)
    prepared = compositing.prepare(original, masks(skin=skin, protect=protect))
    result = np.asarray(compositing.composite(original, png(generated), prepared)).astype(int)
    base = np.asarray(original).astype(int)

    # One brow: where the model drew the moved brow, the original skin is kept, not a dark bar.
    assert result[215:227, 200:400].min() >= base[215:227, 200:400].min() - 5
    # The original brow and lips are untouched.
    assert np.array_equal(result[250:262, 200:400], base[250:262, 200:400])
    assert np.array_equal(result[575:625, 240:360], base[575:625, 240:360])
    # The skin takes the product's effect (brighter) away from the features.
    assert result[450, 300, 0] - base[450, 300, 0] >= 20


def test_lipstick_cannot_make_lips_bigger():
    original = face_like(brow_y=250, lip_half_height=22)
    generated = face_like(brow_y=250, lip_half_height=40)
    lips = np.zeros((900, 600), np.float32)
    import cv2

    cv2.ellipse(lips, (300, 600), (70, 22), 0, 0, 360, 1.0, -1)
    prepared = compositing.prepare(original, masks(makeup=lips))
    result = np.asarray(compositing.composite(original, png(generated), prepared)).astype(int)
    base = np.asarray(original).astype(int)
    # Rows well above the original lip edge stay original skin, though the model drew lip there.
    assert np.array_equal(result[562:572, 260:340], base[562:572, 260:340])
