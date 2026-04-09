import numpy as np
import pytest

from sdimg.image import (
    adjust_brightness_contrast,
    clahe_norm,
    gaussian_blur,
    hist_norm,
    is_image,
    median_blur,
    minmax_norm,
    sharpen,
    to_gray,
    to_rgb,
    to_uint8,
    zscore_norm,
)
from sdimg.image.denoise import denoise


def test_to_uint8_clips_and_rounds() -> None:
    src = np.array([-1.2, 0.49, 0.5, 254.6, 300.0], dtype=np.float32)
    out = to_uint8(src)

    assert out.dtype == np.uint8
    assert out.tolist() == [0, 0, 0, 255, 255]


def test_to_gray_from_rgb_returns_2d_uint8() -> None:
    src = np.array(
        [
            [[10, 20, 30], [40, 50, 60]],
            [[70, 80, 90], [100, 110, 120]],
        ],
        dtype=np.uint8,
    )
    out = to_gray(src)

    assert out.shape == (2, 2)
    assert out.dtype == np.uint8


def test_to_rgb_from_2d_returns_3_channels() -> None:
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = to_rgb(src)

    assert out.shape == (2, 2, 3)
    assert out.dtype == np.uint8
    assert np.array_equal(out[..., 0], src)
    assert np.array_equal(out[..., 1], src)
    assert np.array_equal(out[..., 2], src)


def test_zscore_norm_rejects_non_positive_std_range() -> None:
    with pytest.raises(ValueError, match="std_range must be greater than 0"):
        zscore_norm(np.zeros((8, 8), dtype=np.uint8), std_range=0.0)


def test_minmax_norm_returns_uint8() -> None:
    src = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    out = minmax_norm(src)

    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_denoise_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        denoise(src)


def test_hist_norm_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        hist_norm(src)


def test_gaussian_blur_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        gaussian_blur(src, ksize=(3, 3), sigmaX=1.0)


def test_adjust_brightness_contrast_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        adjust_brightness_contrast(src)


def test_sharpen_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        sharpen(src)


def test_to_gray_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_gray("not-an-array")  # type: ignore[arg-type]


def test_clahe_norm_returns_uint8() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    out = clahe_norm(src)
    assert out.dtype == np.uint8
    assert out.shape == src.shape

    src_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    out_rgb = clahe_norm(src_rgb)
    assert out_rgb.dtype == np.uint8
    assert out_rgb.shape == src_rgb.shape


def test_median_blur_returns_uint8_and_same_shape() -> None:
    src = np.zeros((10, 10, 3), dtype=np.uint8)
    out = median_blur(src, ksize=3)
    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_to_rgb_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_rgb("not-an-array")  # type: ignore[arg-type]


def test_to_uint8_rejects_non_ndarray_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        to_uint8([1, 2, 3])  # type: ignore[arg-type]


def test_clahe_norm_accepts_non_uint8_input() -> None:
    src = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = clahe_norm(src)
    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_hist_norm_accepts_non_uint8_input() -> None:
    src = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    out = hist_norm(src)
    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_denoise_handles_4_channel_image() -> None:
    src = np.zeros((10, 10, 4), dtype=np.uint8)
    out = denoise(src)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == (10, 10)


def test_denoise_handles_2_channel_image() -> None:
    src = np.zeros((10, 10, 2), dtype=np.uint8)
    out = denoise(src)
    assert out.dtype == np.uint8
    assert out.ndim == 2
    assert out.shape == (10, 10)


def test_denoise_returns_uint8_and_same_shape_for_gray_and_rgb() -> None:
    src_gray = np.zeros((10, 10), dtype=np.uint8)
    out_gray = denoise(src_gray)
    assert out_gray.dtype == np.uint8
    assert out_gray.shape == src_gray.shape

    src_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    out_rgb = denoise(src_rgb)
    assert out_rgb.dtype == np.uint8
    assert out_rgb.shape == src_rgb.shape


# --- is_image ---


def test_is_image_returns_true_for_valid_images() -> None:
    assert is_image(np.zeros((4, 4), dtype=np.uint8)) is True
    assert is_image(np.zeros((4, 4, 3), dtype=np.uint8)) is True
    assert is_image(np.zeros((4, 4, 1), dtype=np.float32)) is True


def test_is_image_returns_false_for_invalid_inputs() -> None:
    assert is_image("not-an-array") is False
    assert is_image(np.zeros((2, 3, 4, 5), dtype=np.uint8)) is False
    assert is_image(np.zeros((4, 4, 5), dtype=np.uint8)) is False


# --- adjust_brightness_contrast happy path ---


def test_adjust_brightness_increases_pixel_values() -> None:
    src = np.full((4, 4), 100, dtype=np.uint8)
    out = adjust_brightness_contrast(src, brightness=0.5)
    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert np.all(out > 100)


def test_adjust_contrast_changes_pixel_spread() -> None:
    src = np.array([[80, 120], [80, 120]], dtype=np.uint8)
    out = adjust_brightness_contrast(src, contrast=0.5)
    assert out.dtype == np.uint8
    diff_out = int(out[0, 1]) - int(out[0, 0])
    diff_src = int(src[0, 1]) - int(src[0, 0])
    assert diff_out > diff_src


# --- gaussian_blur happy path ---


def test_gaussian_blur_smooths_image() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[5, 5] = 255
    out = gaussian_blur(src, ksize=(3, 3), sigmaX=1.0)
    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert out[5, 5] < 255


# --- sharpen happy path ---


def test_sharpen_increases_local_contrast() -> None:
    src = np.full((10, 10), 128, dtype=np.uint8)
    src[4:6, 4:6] = 200
    out = sharpen(src, alpha=1.0)
    assert out.dtype == np.uint8
    assert out.shape == src.shape


# --- zscore_norm edge case ---


def test_zscore_norm_constant_image_returns_midgray() -> None:
    src = np.full((8, 8), 50, dtype=np.uint8)
    out = zscore_norm(src)
    assert out.dtype == np.uint8
    assert np.all(out == 128) or np.all(out == 127)


# --- to_rgb edge cases ---


def test_to_rgb_from_4_channel_drops_alpha() -> None:
    src = np.zeros((3, 3, 4), dtype=np.uint8)
    src[..., 0] = 10
    src[..., 1] = 20
    src[..., 2] = 30
    src[..., 3] = 255
    out = to_rgb(src)
    assert out.shape == (3, 3, 3)
    assert np.array_equal(out[..., 0], src[..., 0])


def test_to_rgb_from_1_channel_repeats_gray() -> None:
    src = np.full((3, 3, 1), 42, dtype=np.uint8)
    out = to_rgb(src)
    assert out.shape == (3, 3, 3)
    assert np.all(out == 42)


# --- to_gray edge cases ---


def test_to_gray_from_4_channel_uses_rgb_weights() -> None:
    src = np.zeros((2, 2, 4), dtype=np.uint8)
    src[..., 0] = 255
    out = to_gray(src)
    assert out.shape == (2, 2)
    assert out.dtype == np.uint8
    assert np.all(out > 0)


def test_to_gray_2d_passthrough() -> None:
    src = np.array([[10, 20], [30, 40]], dtype=np.uint8)
    out = to_gray(src)
    assert np.array_equal(out, src)
