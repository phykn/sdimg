import numpy as np
import pytest

from sdimg.image import denoise


def test_denoise_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        denoise(src)


@pytest.mark.parametrize("channels", [2, 4])
def test_denoise_preserves_alpha_channels(channels: int) -> None:
    src = np.zeros((10, 10, channels), dtype=np.uint8)
    src[..., -1] = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = denoise(src)
    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert np.array_equal(out[..., -1], src[..., -1])


def test_denoise_returns_uint8_and_same_shape_for_gray_and_rgb() -> None:
    src_gray = np.zeros((10, 10), dtype=np.uint8)
    out_gray = denoise(src_gray)
    assert out_gray.dtype == np.uint8
    assert out_gray.shape == src_gray.shape

    src_rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    out_rgb = denoise(src_rgb)
    assert out_rgb.dtype == np.uint8
    assert out_rgb.shape == src_rgb.shape
