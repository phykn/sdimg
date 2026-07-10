import base64

import numpy as np
import pytest

import sdimg.image.codec as codec_module
from sdimg.image import decode_image, encode_image


@pytest.mark.parametrize("shape", [(5, 5), (5, 5, 1), (5, 5, 2), (5, 5, 3)])
def test_codec_roundtrips_supported_non_alpha_shapes(shape: tuple[int, ...]) -> None:
    image = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    out = decode_image(encode_image(image))
    expected = image if len(shape) == 2 or shape[-1] == 3 else image[..., 0]
    assert np.array_equal(out, expected)


def test_codec_roundtrips_rgba_losslessly() -> None:
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 0] = 200
    image[..., 3] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    assert np.array_equal(decode_image(encode_image(image)), image)


@pytest.mark.parametrize(
    "payload",
    ["not-base64", base64.b64encode(b"Cbad").decode("ascii")],
)
def test_decode_rejects_malformed_or_legacy_prefix(payload: str) -> None:
    with pytest.raises(ValueError):
        decode_image(payload)


@pytest.mark.parametrize("method", [-1, 7, True])
def test_encode_rejects_invalid_webp_method(method: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_image(np.zeros((2, 2), dtype=np.uint8), method=method)  # type: ignore[arg-type]


@pytest.mark.parametrize("quality", [-1, 101, True])
def test_encode_rejects_invalid_webp_quality(quality: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_image(np.zeros((2, 2), dtype=np.uint8), quality=quality)  # type: ignore[arg-type]


def test_encode_image_wraps_pillow_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_save(*args: object, **kwargs: object) -> None:
        raise OSError("write failed")

    monkeypatch.setattr(codec_module.Image.Image, "save", fail_save)

    with pytest.raises(RuntimeError, match="encode_image failed"):
        encode_image(np.zeros((2, 2), dtype=np.uint8))
