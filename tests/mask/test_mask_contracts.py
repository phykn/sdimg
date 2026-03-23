import numpy as np
import pytest

from sdimg.mask import (
    distance_transform,
    extract_edge,
    convex_hull,
    concave_hull,
    fill_holes,
    get_box_from_mask,
    get_box_from_coords,
    get_box_size,
    get_centroid,
    get_coords,
    get_roi_size,
    morphology,
    pick_largest,
    to_mask,
    to_roi_box,
)


def test_to_mask_accepts_bool_and_returns_uint8_binary() -> None:
    src = np.array([[True, False], [False, True]])
    out = to_mask(src)

    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_to_mask_rejects_non_binary_values() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)


def test_morphology_returns_binary_uint8() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = morphology(src, op="erode", ksize=(3, 3), iterations=1)

    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_get_box_from_mask_uses_wmin_hmin_wmax_hmax_order() -> None:
    src = np.zeros((8, 8), dtype=np.uint8)
    src[2:5, 3:7] = 1

    box = get_box_from_mask(src)

    assert box == (3, 2, 7, 5)


def test_to_roi_returns_none_for_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    assert to_roi_box(src) is None


def test_get_box_from_mask_returns_none_for_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    assert get_box_from_mask(src) is None


def test_get_box_from_coords_returns_none_for_empty_coords() -> None:
    coords = np.empty((0, 2), dtype=np.int64)
    assert get_box_from_coords(coords) is None


def test_to_roi_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [0, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        to_roi_box(src)


def test_get_box_size_returns_area_for_valid_bbox() -> None:
    assert get_box_size((3, 2, 7, 5)) == 12


def test_get_box_size_rejects_invalid_bbox_order() -> None:
    with pytest.raises(ValueError, match="wmin < wmax and hmin < hmax"):
        get_box_size((5, 1, 5, 4))


def test_pick_largest_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        pick_largest(src)


def test_convex_hull_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        convex_hull(src)


def test_extract_edge_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        extract_edge(src)


def test_distance_transform_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        distance_transform(src)


def test_morphology_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        morphology(src, op="open")


def test_get_coords_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        get_coords(src)


def test_get_roi_size_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        get_roi_size(src)


def test_get_centroid_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        get_centroid(src)


def test_get_centroid_returns_none_for_empty_mask() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    assert get_centroid(src) is None


def test_concave_hull_returns_expected_and_handles_empty() -> None:
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert np.array_equal(concave_hull(empty), empty)

    src = np.zeros((10, 10), dtype=np.uint8)
    # L자 모양의 마스크 (concave)
    src[2:8, 2:4] = 1
    src[6:8, 2:8] = 1
    out = concave_hull(src, concavity=2.0)
    assert out.dtype == np.uint8
    assert np.count_nonzero(out) >= np.count_nonzero(src)


def test_fill_holes_handles_empty_single_and_multi_holes() -> None:
    empty = np.zeros((5, 5), dtype=np.uint8)
    assert np.array_equal(fill_holes(empty), empty)

    # 단일 구멍이 있는 마스크
    single_hole = np.zeros((7, 7), dtype=np.uint8)
    single_hole[1:6, 1:6] = 1
    single_hole[3, 3] = 0
    assert np.count_nonzero(single_hole) == 24
    out1 = fill_holes(single_hole)
    assert np.count_nonzero(out1) == 25
    assert out1[3, 3] == 1

    # 다중 구멍이 있는 마스크
    multi_hole = np.zeros((7, 7), dtype=np.uint8)
    multi_hole[1:6, 1:6] = 1
    multi_hole[2, 2] = 0
    multi_hole[4, 4] = 0
    out2 = fill_holes(multi_hole)
    assert np.count_nonzero(out2) == 25
    assert out2[2, 2] == 1
    assert out2[4, 4] == 1


def test_distance_transform_returns_valid_distances() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[2, 2] = 1
    out = distance_transform(src)
    assert out.dtype == np.float32
    assert out.shape == src.shape
    assert out[2, 2] > 0
    assert out[0, 0] <= out[2, 2]


def test_get_coords_transpose_returns_correct_shape() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[1, 1] = 1
    src[2, 3] = 1
    coords_default = get_coords(src)
    assert coords_default.shape == (2, 2)
    
    coords_transposed = get_coords(src, transpose=True)
    assert coords_transposed.shape == (2, 2)
    assert np.array_equal(coords_default.T, coords_transposed)
