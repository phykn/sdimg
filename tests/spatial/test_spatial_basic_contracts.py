import numpy as np
import pytest

from sdimg.spatial import (
    crop,
    flip,
    pad_to_square,
    resize,
    resize_to_long_side,
    rotate,
)


@pytest.mark.parametrize(
    "array",
    [
        np.arange(16, dtype=np.float32).reshape(4, 4),
        np.arange(16, dtype=np.int64).reshape(4, 4),
        np.arange(16).reshape(4, 4) > 7,
    ],
)
def test_resize_preserves_input_dtype(array: np.ndarray) -> None:
    out = resize(array, width=8)
    assert out.dtype == array.dtype
    assert out.shape == (8, 8)


def test_rotate_and_flip_return_independent_contiguous_arrays() -> None:
    array = np.arange(12, dtype=np.uint8).reshape(3, 4)
    for out in (rotate(array, degrees=0), flip(array, "horizontal")):
        assert out.flags.c_contiguous
        assert not np.shares_memory(out, array)


def test_crop_validates_xy_bbox_and_returns_copy() -> None:
    array = np.arange(20, dtype=np.int16).reshape(4, 5)
    out = crop(array, (1, 1, 4, 3))
    assert out.dtype == array.dtype
    assert out.shape == (2, 3)
    assert np.array_equal(out, array[1:3, 1:4])
    assert not np.shares_memory(out, array)


def test_pad_to_square_returns_original_bbox() -> None:
    array = np.ones((2, 4), dtype=np.uint16)
    out, bbox = pad_to_square(array, return_bbox=True)
    assert out.shape == (4, 4)
    assert out.dtype == array.dtype
    assert bbox == (0, 0, 4, 2)


def test_resize_to_long_side_preserves_aspect_ratio() -> None:
    array = np.zeros((10, 20, 3), dtype=np.uint8)
    assert resize_to_long_side(array, 40).shape == (20, 40, 3)


def test_resize_preserves_singleton_channel_axis() -> None:
    array = np.zeros((4, 5, 1), dtype=np.uint8)
    assert resize(array, height=8, width=10).shape == (8, 10, 1)


def test_resize_to_long_side_preserves_singleton_channel_axis() -> None:
    array = np.zeros((4, 5, 1), dtype=np.uint8)
    assert resize_to_long_side(array, 10).shape == (8, 10, 1)


@pytest.mark.parametrize("width", [True, 2.5, 0])
def test_resize_rejects_non_positive_integer_dimensions(width: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        resize(np.zeros((2, 2), dtype=np.uint8), width=width)  # type: ignore[arg-type]


def test_spatial_operations_reject_empty_source() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resize_to_long_side(np.zeros((0, 2), dtype=np.uint8), 4)


@pytest.mark.parametrize("degrees", [-90, 45, 360, True])
def test_rotate_rejects_invalid_degrees(degrees: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        rotate(np.zeros((2, 3), dtype=np.uint8), degrees=degrees)  # type: ignore[arg-type]
