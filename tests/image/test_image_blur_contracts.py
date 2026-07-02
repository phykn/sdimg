import numpy as np
import pytest

from sdimg.image import gaussian_blur, median_blur


def test_gaussian_blur_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        gaussian_blur(src, ksize=(3, 3), sigmaX=1.0)


def test_median_blur_returns_uint8_and_same_shape() -> None:
    src = np.zeros((10, 10, 3), dtype=np.uint8)
    out = median_blur(src, ksize=3)
    assert out.dtype == np.uint8
    assert out.shape == src.shape


def test_gaussian_blur_smooths_image() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[5, 5] = 255
    out = gaussian_blur(src, ksize=(3, 3), sigmaX=1.0)
    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert out[5, 5] < 255
