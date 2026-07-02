from collections.abc import Callable

import numpy as np
import pytest

from sdimg.mask import (
    get_box_from_coords,
    get_box_from_mask,
    get_box_size,
    get_centroid,
    get_coords,
    get_roi_size,
    to_roi_box,
)


def test_get_box_from_mask_uses_wmin_hmin_wmax_hmax_order() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[2:5, 3:7] = 1

    box = get_box_from_mask(src)

    assert box == (3, 2, 7, 5)


@pytest.mark.parametrize(
    "func,src",
    [
        (to_roi_box, np.zeros((5, 5), dtype=np.uint8)),
        (get_box_from_mask, np.zeros((5, 5), dtype=np.uint8)),
        (get_box_from_coords, np.empty((0, 2), dtype=np.int64)),
        (get_centroid, np.zeros((5, 5), dtype=np.uint8)),
    ],
)
def test_empty_inputs_return_none(
    func: Callable[[np.ndarray], object],
    src: np.ndarray,
) -> None:
    assert func(src) is None


def test_get_box_size_returns_area_for_valid_bbox() -> None:
    assert get_box_size((3, 2, 7, 5)) == 12


def test_get_box_size_rejects_invalid_bbox_order() -> None:
    with pytest.raises(ValueError, match="wmin < wmax and hmin < hmax"):
        get_box_size((5, 1, 5, 4))


@pytest.mark.parametrize(
    "func",
    [to_roi_box, get_coords, get_roi_size, get_centroid],
)
def test_mask_bbox_helpers_reject_non_binary_mask(
    func: Callable[[np.ndarray], object],
) -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        func(src)


def test_get_coords_transpose_returns_correct_shape() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[1, 1] = 1
    src[2, 3] = 1
    coords_default = get_coords(src)
    assert coords_default.shape == (2, 2)

    coords_transposed = get_coords(src, transpose=True)
    assert coords_transposed.shape == (2, 2)
    assert np.array_equal(coords_default.T, coords_transposed)


def test_get_centroid_returns_correct_center() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    result = get_centroid(src)
    assert result is not None
    assert abs(result[0] - 3.0) < 1e-6
    assert abs(result[1] - 3.0) < 1e-6


def test_get_roi_size_returns_correct_count() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[1:3, 1:4] = 1
    assert get_roi_size(src) == 6


def test_get_roi_size_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    assert get_roi_size(src) == 0


def test_to_roi_box_returns_correct_roi_and_box() -> None:
    src = np.zeros((6, 6), dtype=np.uint8)
    src[1:4, 2:5] = 1
    result = to_roi_box(src)
    assert result is not None
    assert result["box"] == (2, 1, 5, 4)
    roi = result["roi"]
    assert roi.shape == (3, 3)
    assert np.all(roi == 1)


def test_get_box_from_coords_returns_correct_box() -> None:
    coords = np.array([[1, 2], [3, 5]], dtype=np.int64)
    box = get_box_from_coords(coords)
    assert box == (2, 1, 6, 4)


def test_get_box_from_coords_single_point() -> None:
    coords = np.array([[3, 4]], dtype=np.int64)
    box = get_box_from_coords(coords)
    assert box == (4, 3, 5, 4)


def test_get_box_from_coords_rejects_invalid_shape() -> None:
    coords = np.array([1, 2, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="coords must have shape"):
        get_box_from_coords(coords)
