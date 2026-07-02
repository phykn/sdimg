import numpy as np
import pytest

from sdimg.mask import pick_largest


def test_pick_largest_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        pick_largest(src)


def test_pick_largest_selects_bigger_component() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[0:2, 0:2] = 1
    src[5:9, 5:9] = 1
    out = pick_largest(src)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) == 16
    assert out[6, 6] == 1
    assert out[0, 0] == 0


def test_pick_largest_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = pick_largest(src)
    assert np.count_nonzero(out) == 0
