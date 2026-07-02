import numpy as np
import pytest

from sdimg.mask import distance_transform


def test_distance_transform_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        distance_transform(src)


def test_distance_transform_returns_valid_distances() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[2, 2] = 1
    out = distance_transform(src)
    assert out.dtype == np.float32
    assert out.shape == src.shape
    assert out[2, 2] > 0
    assert out[0, 0] <= out[2, 2]


def test_distance_transform_l1_returns_float32() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = distance_transform(src, distance_type="l1")
    assert out.dtype == np.float32
    assert out[3, 3] > out[2, 2]


def test_distance_transform_c_returns_float32() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = distance_transform(src, distance_type="c")
    assert out.dtype == np.float32
    assert out[3, 3] > 0


def test_distance_transform_rejects_invalid_type() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="distance_type must be one of"):
        distance_transform(src, distance_type="invalid")
