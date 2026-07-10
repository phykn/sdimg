import numpy as np
import pytest

from sdimg.mask import (
    apply_morphology,
    compute_distance_transform,
    extract_boundary,
    fill_concave_hull,
    fill_convex_hull,
    fill_holes,
    keep_largest_component,
)


def test_fill_holes_preserves_exterior_when_object_touches_corner() -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[:4, :4] = 1
    mask[1, 1] = 0
    out = fill_holes(mask)
    assert out[1, 1] == 1
    assert np.all(out[4:, :] == 0)
    assert np.all(out[:, 4:] == 0)


@pytest.mark.parametrize("function", [fill_convex_hull, fill_concave_hull])
def test_hulls_preserve_disconnected_components(function: object) -> None:
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[1:4, 1:3] = 1
    mask[8:11, 13:15] = 1
    out = function(mask)  # type: ignore[operator]
    assert out[4:8, 5:11].sum() == 0
    assert np.all(out[mask == 1] == 1)


def test_morphology_large_kernel_uses_zero_exterior() -> None:
    mask = np.ones((9, 9), dtype=np.uint8)
    out = apply_morphology(mask, "erode", kernel_size=(5, 5), iterations=2)
    assert np.count_nonzero(out) == 1
    assert out[4, 4] == 1


def test_extract_boundary_supports_large_kernel() -> None:
    mask = np.ones((7, 7), dtype=np.uint8)
    out = extract_boundary(mask, kernel_size=(5, 5))
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert out[3, 3] == 0
    assert out[0, 0] == 1


def test_keep_largest_component_keeps_only_largest() -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1, 1] = 1
    mask[4:7, 4:7] = 1
    out = keep_largest_component(mask)
    assert out[1, 1] == 0
    assert out[4:7, 4:7].sum() == 9


@pytest.mark.parametrize("connectivity", [0, 5, 9])
def test_keep_largest_component_rejects_invalid_connectivity(connectivity: int) -> None:
    with pytest.raises(ValueError):
        keep_largest_component(np.zeros((3, 3), dtype=np.uint8), connectivity)


@pytest.mark.parametrize(
    "distance_type,mask_size,valid",
    [
        ("l1", 3, True),
        ("l1", 5, False),
        ("c", 3, True),
        ("c", 5, False),
        ("l2", 3, True),
        ("l2", 5, True),
    ],
)
def test_distance_type_and_mask_size_contract(
    distance_type: str,
    mask_size: int,
    valid: bool,
) -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    if valid:
        out = compute_distance_transform(mask, distance_type, mask_size)
        assert out.dtype == np.float32
        assert out[2, 2] > out[1, 1]
    else:
        with pytest.raises(ValueError):
            compute_distance_transform(mask, distance_type, mask_size)


def test_mask_processing_empty_outputs_keep_contracts() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    for function in (
        keep_largest_component,
        fill_holes,
        fill_convex_hull,
        fill_concave_hull,
        extract_boundary,
    ):
        out = function(mask)
        assert out.dtype == np.uint8
        assert np.count_nonzero(out) == 0
    distance = compute_distance_transform(mask)
    assert distance.dtype == np.float32
    assert np.count_nonzero(distance) == 0
