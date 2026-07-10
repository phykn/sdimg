import numpy as np
import pytest

from sdimg.mask import fill_concave_hull, fill_convex_hull


def test_fill_convex_hull_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        fill_convex_hull(src)


def test_fill_concave_hull_returns_expected_and_handles_empty() -> None:
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert np.array_equal(fill_concave_hull(empty), empty)

    src = np.zeros((10, 10), dtype=np.uint8)
    # L자 모양의 마스크 (concave)
    src[2:8, 2:4] = 1
    src[6:8, 2:8] = 1
    out = fill_concave_hull(src, concavity=2.0)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) >= np.count_nonzero(src)


def test_fill_convex_hull_fills_concave_region() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[2:8, 2:4] = 1
    src[6:8, 2:8] = 1
    out = fill_convex_hull(src)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert np.count_nonzero(out) >= np.count_nonzero(src)


def test_fill_convex_hull_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = fill_convex_hull(src)
    assert np.count_nonzero(out) == 0


def test_fill_concave_hull_single_pixel_returns_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[2, 2] = 1
    out = fill_concave_hull(src)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) >= 1
