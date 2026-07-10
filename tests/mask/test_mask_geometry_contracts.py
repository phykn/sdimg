import numpy as np
import pytest

from sdimg.mask import (
    count_foreground,
    extract_roi,
    find_bbox,
    find_bbox_from_points,
    find_centroid,
    find_foreground_points,
    measure_bbox_area,
)


def _asymmetric_mask() -> np.ndarray:
    mask = np.zeros((7, 10), dtype=np.uint8)
    mask[2:5, 4:9] = 1
    return mask


def test_geometry_uses_xy_order() -> None:
    mask = _asymmetric_mask()
    points = find_foreground_points(mask)
    assert points.shape == (15, 2)
    assert points[0].tolist() == [4, 2]
    assert find_bbox(mask) == (4, 2, 9, 5)
    assert find_centroid(mask) == (6.0, 3.0)


def test_find_bbox_from_xy_points() -> None:
    points = np.array([[4, 2], [8, 4]], dtype=np.int64)
    assert find_bbox_from_points(points) == (4, 2, 9, 5)


def test_extract_roi_returns_tuple_and_independent_mask() -> None:
    mask = _asymmetric_mask()
    result = extract_roi(mask)
    assert result is not None
    roi, bbox = result
    assert bbox == (4, 2, 9, 5)
    assert roi.shape == (3, 5)
    assert not np.shares_memory(roi, mask)


def test_measurements_and_empty_geometry() -> None:
    mask = _asymmetric_mask()
    assert count_foreground(mask) == 15
    assert measure_bbox_area((4, 2, 9, 5)) == 15
    assert measure_bbox_area(None) == 0

    empty = np.zeros((3, 4), dtype=np.uint8)
    assert find_foreground_points(empty).shape == (0, 2)
    assert find_bbox(empty) is None
    assert find_centroid(empty) is None
    assert extract_roi(empty) is None


@pytest.mark.parametrize(
    "points",
    [
        np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        np.array([[1.5, 2.0]], dtype=np.float32),
        np.array([[-1, 2]], dtype=np.int64),
        np.array([[True, False]], dtype=np.bool_),
    ],
)
def test_find_bbox_from_points_rejects_invalid_points(points: np.ndarray) -> None:
    with pytest.raises(ValueError):
        find_bbox_from_points(points)


def test_measure_bbox_area_validates_type_and_order() -> None:
    with pytest.raises(TypeError):
        measure_bbox_area((0.0, 0, 2, 2))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        measure_bbox_area((2, 0, 1, 2))
