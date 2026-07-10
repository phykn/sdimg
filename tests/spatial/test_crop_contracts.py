import numpy as np
import pytest

from sdimg.spatial import crop


def test_crop_rejects_invalid_bbox() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="bbox is outside array bounds"):
        crop(src, bbox=(0, 0, 6, 4))


def test_crop_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        crop(src, bbox=(0, 0, 1, 1))


def test_crop_returns_independent_copy() -> None:
    src = np.arange(25, dtype=np.uint8).reshape(5, 5)
    cropped = crop(src, bbox=(1, 1, 3, 3))
    cropped[0, 0] = 255

    assert src[1, 1] != 255


def test_crop_returns_correct_content() -> None:
    src = np.arange(25, dtype=np.uint8).reshape(5, 5)
    out = crop(src, bbox=(1, 2, 4, 4))
    assert out.shape == (2, 3)
    assert out[0, 0] == src[2, 1]
    assert out[1, 2] == src[3, 3]
