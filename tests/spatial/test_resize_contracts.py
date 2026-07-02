import numpy as np
import pytest

from sdimg.spatial import resize, resize_keep_ratio


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
    with pytest.raises(ValueError, match="src must have shape"):
        resize(src, width=10)


def test_resize_keep_ratio_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="src must have shape"):
        resize_keep_ratio(src, long_side=10)


def test_resize_keep_ratio_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize_keep_ratio(src, long_side=40)
    assert out.shape == (20, 40, 3)


def test_resize_keep_ratio_scales_down() -> None:
    src = np.zeros((100, 200), dtype=np.uint8)
    out = resize_keep_ratio(src, long_side=100)
    assert out.shape == (50, 100)


def test_resize_keep_ratio_rejects_non_positive_long_side() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    with pytest.raises(ValueError, match="long_side must be greater than 0"):
        resize_keep_ratio(src, long_side=0)


def test_resize_with_both_height_and_width() -> None:
    src = np.zeros((10, 20), dtype=np.uint8)
    out = resize(src, height=5, width=10)
    assert out.shape == (5, 10)


def test_resize_with_only_height_preserves_aspect_ratio() -> None:
    src = np.zeros((10, 20, 3), dtype=np.uint8)
    out = resize(src, height=20)
    assert out.shape[:2] == (20, 40)
