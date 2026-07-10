import numpy as np
import pytest

from sdimg.image import sharpen


def test_sharpen_rejects_invalid_image_shape() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="image must have shape"):
        sharpen(src)


def test_sharpen_increases_local_contrast() -> None:
    src = np.full((10, 10), 128, dtype=np.uint8)
    src[4:6, 4:6] = 200
    out = sharpen(src, amount=1.0)
    assert out.dtype == np.uint8
    assert out.shape == src.shape
