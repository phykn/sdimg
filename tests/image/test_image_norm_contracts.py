import numpy as np
import pytest

from sdimg.image import clahe_norm, hist_norm, minmax_norm, zscore_norm


def test_zscore_norm_rejects_non_positive_std_range() -> None:
    with pytest.raises(ValueError, match="std_range must be greater than 0"):
        zscore_norm(np.zeros((8, 8), dtype=np.uint8), std_range=0.0)


def test_minmax_norm_returns_uint8() -> None:
    src = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    out = minmax_norm(src)

    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_hist_norm_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        hist_norm(src)


def test_clahe_norm_returns_uint8() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    out = clahe_norm(src)
    assert out.dtype == np.uint8
    assert out.shape == src.shape

    src_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    out_rgb = clahe_norm(src_rgb)
    assert out_rgb.dtype == np.uint8
    assert out_rgb.shape == src_rgb.shape


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


def test_zscore_norm_constant_image_returns_midgray() -> None:
    src = np.full((8, 8), 50, dtype=np.uint8)
    out = zscore_norm(src)
    assert out.dtype == np.uint8
    assert np.all(out == 128) or np.all(out == 127)
