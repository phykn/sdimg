from collections.abc import Callable

import numpy as np
import pytest

from sdimg.image import convert_to_gray, convert_to_rgb, convert_to_uint8, is_image


def test_convert_to_uint8_clips_and_rounds() -> None:
    src = np.array([-1.2, 0.49, 0.5, 254.6, 300.0], dtype=np.float32)
    out = convert_to_uint8(src)

    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 0, 255, 255]

    int_src = np.array([-5, 0, 255, 300], dtype=np.int16)
    assert convert_to_uint8(int_src).tolist() == [0, 0, 255, 255]


def test_convert_to_gray_from_rgb_returns_2d_uint8() -> None:
    src = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    out = convert_to_gray(src)

    assert out.shape == (2, 2)
    assert out.dtype == np.uint8


def test_convert_to_rgb_from_2d_returns_3_channels() -> None:
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = convert_to_rgb(src)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 0], src)
    assert np.array_equal(out[..., 1], src)
    assert np.array_equal(out[..., 2], src)


@pytest.mark.parametrize(
    "func,bad_value",
    [
        (convert_to_gray, "not-an-array"),
        (convert_to_rgb, "not-an-array"),
        (convert_to_uint8, [1, 2, 3]),
    ],
)
def test_image_convert_functions_reject_non_ndarray_input(
    func: Callable[[object], object],
    bad_value: object,
) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        func(bad_value)


def test_is_image_returns_true_for_valid_images() -> None:
    assert is_image(np.zeros((4, 4), dtype=np.uint8)) is True
    assert is_image(np.zeros((4, 4, 3), dtype=np.uint8)) is True
    assert is_image(np.zeros((4, 4, 2), dtype=np.uint8)) is True
    assert is_image(np.zeros((4, 4, 1), dtype=np.float32)) is True


def test_is_image_returns_false_for_invalid_inputs() -> None:
    assert is_image("not-an-array") is False
    assert is_image(np.zeros((2, 3, 4, 5), dtype=np.uint8)) is False
    assert is_image(np.zeros((4, 4, 5), dtype=np.uint8)) is False


def test_convert_to_rgb_from_4_channel_drops_alpha() -> None:
    src = np.zeros((3, 3, 4), dtype=np.uint8)
    src[..., 0] = 10
    src[..., 1] = 20
    src[..., 2] = 30
    src[..., 3] = 255
    out = convert_to_rgb(src)
    assert out.shape == (3, 3, 3)
    assert np.array_equal(out[..., 0], src[..., 0])


def test_convert_to_rgb_from_1_channel_repeats_gray() -> None:
    src = np.full((3, 3, 1), 42, dtype=np.uint8)
    out = convert_to_rgb(src)
    assert out.shape == (3, 3, 3)
    assert np.all(out == 42)


def test_convert_to_rgb_from_2_channel_ignores_alpha() -> None:
    gray = np.arange(9, dtype=np.uint8).reshape(3, 3)
    src = np.stack([gray, np.full_like(gray, 255)], axis=2)

    out = convert_to_rgb(src)

    assert out.shape == (3, 3, 3)
    assert np.array_equal(out[..., 0], gray)
    assert np.array_equal(out[..., 1], gray)
    assert np.array_equal(out[..., 2], gray)


def test_convert_to_gray_from_4_channel_uses_rgb_weights() -> None:
    src = np.zeros((2, 2, 4), dtype=np.uint8)
    src[..., 0] = 255
    out = convert_to_gray(src)
    assert out.shape == (2, 2)
    assert out.dtype == np.uint8
    assert np.all(out > 0)


def test_convert_to_gray_from_2_channel_ignores_alpha() -> None:
    gray = np.arange(9, dtype=np.uint8).reshape(3, 3)
    src = np.stack([gray, np.full_like(gray, 255)], axis=2)

    out = convert_to_gray(src)

    assert np.array_equal(out, gray)


def test_convert_to_gray_2d_passthrough() -> None:
    src = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = convert_to_gray(src)
    assert np.array_equal(out, src)
