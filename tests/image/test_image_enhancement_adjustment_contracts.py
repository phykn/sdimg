import numpy as np
import pytest

from sdimg.image import adjust_brightness_contrast


def test_adjust_brightness_contrast_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        adjust_brightness_contrast(src)


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
