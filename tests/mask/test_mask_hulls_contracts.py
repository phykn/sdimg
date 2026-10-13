from collections.abc import Callable

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


@pytest.mark.parametrize("function", [fill_convex_hull, fill_concave_hull])
def test_hulls_fill_each_component_at_its_original_position(
    function: Callable[[np.ndarray], np.ndarray],
) -> None:
    component = np.zeros((5, 6), dtype=np.uint8)
    component[:, :2] = 1
    component[-2:, :] = 1
    filled = function(component)
    mask = np.zeros((17, 23), dtype=np.uint8)
    expected = np.zeros_like(mask)
    for y, x in [(0, 0), (6, 8), (12, 17)]:
        mask[y:y + 5, x:x + 6] = component
        expected[y:y + 5, x:x + 6] = filled
    mask[0, 22] = expected[0, 22] = 1
    mask[16, :2] = expected[16, :2] = 1
    original = mask.copy()

    out = function(mask)

    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(mask, original)
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    assert not np.shares_memory(out, mask)


@pytest.mark.parametrize("function", [fill_convex_hull, fill_concave_hull])
def test_hulls_union_components_with_nested_bounding_boxes(
    function: Callable[[np.ndarray], np.ndarray],
) -> None:
    outer = np.zeros((19, 21), dtype=np.uint8)
    outer[1:18, 1] = 1
    outer[[1, 17], 1:20] = 1
    inner = np.zeros_like(outer)
    inner[6:12, 7] = 1
    inner[11, 7:14] = 1
    mask = np.maximum(outer, inner)
    expected = np.maximum(function(outer), function(inner))

    np.testing.assert_array_equal(function(mask), expected)


def test_concave_hull_preserves_source_coordinate_tie_breaking() -> None:
    mask = np.zeros((8, 26), dtype=np.uint8)
    mask[[4, 5, 5, 6], [23, 23, 24, 22]] = 1
    expected = mask.copy()
    expected[6, 23] = 1

    np.testing.assert_array_equal(fill_concave_hull(mask), expected)
