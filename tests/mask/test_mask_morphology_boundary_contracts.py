import numpy as np
import pytest

from sdimg.mask import extract_boundary


def test_extract_boundary_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        extract_boundary(src)


def test_extract_boundary_returns_boundary_pixels() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = extract_boundary(src)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert out[3, 3] == 0
    assert out[2, 2] == 1


def test_extract_boundary_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = extract_boundary(src)
    assert np.count_nonzero(out) == 0
