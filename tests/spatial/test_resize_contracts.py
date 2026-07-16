import cv2
import numpy as np
import pytest

from sdimg.spatial import resize, resize_to_long_side


@pytest.mark.parametrize(
    "dtype,value",
    [
        (np.int64, -(2**53) - 1),
        (np.int64, 2**53 + 1),
        (np.uint64, 2**53 + 1),
    ],
)
def test_resize_rejects_integers_outside_float64_exact_range(
    dtype: np.dtype,
    value: int,
) -> None:
    src = np.full((2, 2), value, dtype=dtype)

    with pytest.raises(ValueError, match="float64-backed spatial operations"):
        resize(src, height=2, width=2, interpolation=cv2.INTER_NEAREST)


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_resize_accepts_float64_exact_integer_boundary(dtype: np.dtype) -> None:
    src = np.array([[0, 2**53]], dtype=dtype)

    out = resize(src, height=2, width=4, interpolation=cv2.INTER_NEAREST)

    assert out.dtype == src.dtype
    assert np.array_equal(out[:, ::2], np.repeat(src, 2, axis=0))


def test_resize_with_only_width_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize(src, width=40)

    assert out.shape[:2] == (20, 40)


def test_resize_requires_height_or_width() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="height or width must be provided"):
        resize(src)


def test_resize_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        resize(src, width=10)


def test_resize_to_long_side_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        resize_to_long_side(src, long_side=10)


def test_resize_to_long_side_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize_to_long_side(src, long_side=40)
    assert out.shape == (20, 40, 3)


def test_resize_to_long_side_scales_down() -> None:
    src = np.zeros((100, 200), dtype=np.uint8)
    out = resize_to_long_side(src, long_side=100)
    assert out.shape == (50, 100)


def test_resize_to_long_side_rejects_non_positive_long_side() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="long_side must be greater than 0"):
        resize_to_long_side(src, long_side=0)


def test_resize_with_both_height_and_width() -> None:
    src = np.zeros((10, 20), dtype=np.uint8)
    out = resize(src, height=5, width=10)
    assert out.shape == (5, 10)


def test_resize_with_only_height_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize(src, height=20)
    assert out.shape[:2] == (20, 40)
