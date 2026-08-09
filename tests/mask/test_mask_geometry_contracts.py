import cv2
import numpy as np
import pytest

import sdimg.mask.geometry as geometry

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


def test_geometry_handles_noncontiguous_0255_mask() -> None:
    source = np.zeros((10, 14), dtype=np.uint8)
    mask = source[::2, ::2]
    mask[1:4, 2:6] = 255
    assert not mask.flags.c_contiguous

    assert find_bbox(mask) == (2, 1, 6, 4)
    assert find_centroid(mask) == (3.5, 2.0)
    result = extract_roi(mask)
    assert result is not None
    roi, bbox = result
    assert bbox == (2, 1, 6, 4)
    assert np.array_equal(roi, np.ones((3, 4), dtype=np.uint8))
    assert not np.shares_memory(roi, mask)


def test_dense_geometry_uses_full_mask_extent() -> None:
    mask = np.ones((256, 320), dtype=np.uint8)

    assert find_bbox(mask) == (0, 0, 320, 256)
    assert find_centroid(mask) == (159.5, 127.5)


def test_extract_roi_converts_mask_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    original = geometry.convert_to_mask

    def count_calls(mask: object) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(mask)

    monkeypatch.setattr(geometry, "convert_to_mask", count_calls)

    result = extract_roi(_asymmetric_mask())

    assert result is not None
    assert calls == 1


def test_geometry_wraps_opencv_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> object:
        raise cv2.error("forced failure")

    mask = _asymmetric_mask()
    monkeypatch.setattr(cv2, "boundingRect", fail)
    with pytest.raises(RuntimeError, match="find_bbox failed: forced failure"):
        find_bbox(mask)

    monkeypatch.setattr(cv2, "moments", fail)
    with pytest.raises(RuntimeError, match="find_centroid failed: forced failure"):
        find_centroid(mask)
