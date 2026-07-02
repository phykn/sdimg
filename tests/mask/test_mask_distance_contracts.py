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


@pytest.mark.parametrize("distance_type", ["l1", "c"])
def test_distance_transform_supports_distance_types(distance_type: str) -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = distance_transform(src, distance_type=distance_type)  # type: ignore[arg-type]
    assert out.dtype == np.float32
    assert out[3, 3] > out[2, 2]


def test_distance_transform_empty_mask_returns_zeros() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = distance_transform(src)
    assert out.dtype == np.float32
    assert np.array_equal(out, np.zeros_like(src, dtype=np.float32))


def test_distance_transform_rejects_invalid_type() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="distance_type must be one of"):
        distance_transform(src, distance_type="invalid")
