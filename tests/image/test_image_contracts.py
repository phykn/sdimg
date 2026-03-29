import numpy as np
import pytest

from sdimg.image import (
    adjust_brightness_contrast,
    clahe_norm,
    gaussian_blur,
    hist_norm,
    median_blur,
    minmax_norm,
    sharpen,
    to_gray,
    to_rgb,
    to_uint8,
    zscore_norm,
)
from sdimg.image.denoise import denoise, destripe


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


@pytest.mark.requires_torch
def test_destripe_requires_torch_dependency() -> None:
    pytest.importorskip("torch")

    src = np.zeros((8, 8), dtype=np.uint8)
    out = destripe(src, iterations=1, n_tiles=1, verbose=False)

    assert out.shape == src.shape
    assert out.dtype == np.uint8


@pytest.mark.requires_torch
def test_destripe_wraps_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    torch = pytest.importorskip("torch")
    denoise_module = importlib.import_module("sdimg.image.denoise")

    class _FailingRemover:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def process(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise RuntimeError("torch boom")

        def process_tiled(self, *args: object, **kwargs: object) -> torch.Tensor:
            raise RuntimeError("torch boom")

    monkeypatch.setattr(
        denoise_module,
        "UniversalStripeRemover",
        _FailingRemover,
    )

    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="destripe failed"):
        destripe(src, iterations=1, n_tiles=1, verbose=False)


def test_destripe_rejects_non_positive_iterations() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="iterations must be greater than 0"):
        destripe(src, iterations=0)


def test_destripe_rejects_non_positive_n_tiles() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="n_tiles must be greater than 0"):
        destripe(src, n_tiles=0)


def test_denoise_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        denoise(src)


def test_destripe_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        destripe(src, iterations=1)


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
    with pytest.raises(ValueError, match="numpy.ndarray"):
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


def test_denoise_returns_uint8_and_same_shape_for_gray_and_rgb() -> None:
    src_gray = np.zeros((10, 10), dtype=np.uint8)
    out_gray = denoise(src_gray)
    assert out_gray.dtype == np.uint8
    assert out_gray.shape == src_gray.shape

    src_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    out_rgb = denoise(src_rgb)
    assert out_rgb.dtype == np.uint8
    assert out_rgb.shape == src_rgb.shape
