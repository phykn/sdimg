import numpy as np
import pytest

from sdimg.image import (
    convert_to_gray,
    convert_to_rgb,
    convert_to_uint8,
    is_image,
)


def test_convert_to_uint8_clips_and_rounds_general_array() -> None:
    array = np.array([-1.2, 0.49, 0.5, 254.6, 300.0], dtype=np.float32)
    assert convert_to_uint8(array).tolist() == [0, 0, 0, 255, 255]


def test_convert_to_gray_uses_rgb_and_drops_alpha() -> None:
    image = np.zeros((2, 3, 4), dtype=np.float32)
    image[..., 0] = 255.0
    image[..., 3] = 17.0
    out = convert_to_gray(image)
    assert out.shape == (2, 3)
    assert out.dtype == np.uint8
    assert np.all(out == 76)


def test_convert_to_rgb_uses_gray_and_drops_alpha() -> None:
    gray = np.arange(6, dtype=np.float32).reshape(2, 3)
    image = np.stack([gray, np.full_like(gray, 200.0)], axis=2)
    out = convert_to_rgb(image)
    assert out.shape == (2, 3, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 0], gray.astype(np.uint8))
    assert np.array_equal(out[..., 0], out[..., 1])


@pytest.mark.parametrize(
    "image,expected",
    [
        (np.zeros((2, 2), dtype=np.uint8), True),
        (np.zeros((2, 2, 4), dtype=np.float32), True),
        (np.zeros((0, 2), dtype=np.uint8), False),
        (np.zeros((2, 2), dtype=object), False),
        (np.zeros((2, 2), dtype=np.complex64), False),
        (np.zeros((2, 2, 5), dtype=np.uint8), False),
    ],
)
def test_is_image_reflects_full_image_contract(
    image: np.ndarray, expected: bool
) -> None:
    assert is_image(image) is expected


@pytest.mark.parametrize("function", [convert_to_gray, convert_to_rgb])
def test_color_conversion_rejects_non_array(function: object) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        function("not-array")  # type: ignore[operator]
