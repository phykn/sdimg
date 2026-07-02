import base64

import numpy as np
import pytest

from sdimg.image import decode, encode


def test_encode_decode_round_trips_2d_grayscale() -> None:
    image = np.arange(100, dtype=np.uint8).reshape(10, 10)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_decode_round_trips_rgb() -> None:
    image = np.random.default_rng(0).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_decode_round_trips_rgba() -> None:
    image = np.zeros((8, 8, 4), dtype=np.uint8)
    image[..., 0] = 255
    image[..., 3] = np.arange(64, dtype=np.uint8).reshape(8, 8)

    out = decode(encode(image))

    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out, image)


def test_encode_accepts_single_channel_3d_and_decodes_to_2d() -> None:
    image = np.arange(25, dtype=np.uint8).reshape(5, 5, 1)

    out = decode(encode(image))

    assert out.shape == (5, 5)
    assert np.array_equal(out, image[..., 0])


def test_encode_accepts_two_channel_grayscale_alpha_and_ignores_alpha() -> None:
    gray = np.arange(25, dtype=np.uint8).reshape(5, 5)
    image = np.stack([gray, np.full_like(gray, 17)], axis=2)

    out = decode(encode(image))

    assert out.shape == (5, 5)
    assert np.array_equal(out, gray)


def test_encode_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        encode("not-an-array")  # type: ignore[arg-type]


def test_encode_rejects_non_uint8() -> None:
    with pytest.raises(ValueError, match="uint8"):
        encode(np.zeros((4, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "image",
    [
        np.zeros((2, 3, 4, 1), dtype=np.uint8),
        np.zeros((4, 4, 5), dtype=np.uint8),
    ],
)
def test_encode_rejects_unsupported_shapes(image: np.ndarray) -> None:
    with pytest.raises(ValueError, match="shape"):
        encode(image)


def test_decode_rejects_non_string_input() -> None:
    with pytest.raises(TypeError, match="str"):
        decode(b"not-a-string")  # type: ignore[arg-type]


def test_decode_rejects_invalid_base64() -> None:
    with pytest.raises(ValueError, match="failed to deserialize array"):
        decode("not-base64")


def test_decode_rejects_invalid_prefix() -> None:
    encoded = base64.b64encode(b"Xabc").decode("utf-8")

    with pytest.raises(ValueError, match="invalid payload prefix"):
        decode(encoded)


def test_decode_rejects_invalid_webp_payload() -> None:
    encoded = base64.b64encode(b"Rnot-webp").decode("utf-8")

    with pytest.raises(ValueError, match="failed to deserialize array"):
        decode(encoded)
