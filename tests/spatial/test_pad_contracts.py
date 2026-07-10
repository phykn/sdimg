import numpy as np
import pytest

from sdimg.spatial import pad_to_square


def test_pad_to_square_returns_box_and_square_shape() -> None:
    src = np.ones((3, 5), dtype=np.uint8)
    padded, box = pad_to_square(src, return_bbox=True)

    assert padded.shape == (5, 5)
    assert box == (0, 0, 5, 3)


def test_pad_to_square_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        pad_to_square(src)


def test_pad_to_square_already_square() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    out = pad_to_square(src)
    assert out.shape == (5, 5)
    assert np.array_equal(out, src)


def test_pad_to_square_tall_image() -> None:
    src = np.ones((5, 3), dtype=np.uint8)
    out = pad_to_square(src)
    assert out.shape == (5, 5)
