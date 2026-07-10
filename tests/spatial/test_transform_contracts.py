import numpy as np
import pytest

from sdimg.spatial import flip, rotate


def test_rotate_and_flip_contract() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    rotated = rotate(src, degrees=90)
    flipped = flip(src, direction="horizontal")

    assert rotated.shape == (3, 2)
    assert flipped.shape == src.shape


def test_rotate_rejects_invalid_angle() -> None:
    src = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="degrees must be one of"):
        rotate(src, degrees=45)


def test_rotate_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        rotate(src, degrees=90)


def test_flip_rejects_invalid_src_ndim() -> None:
    src = np.zeros((2, 3, 4, 5), dtype=np.uint8)
    with pytest.raises(ValueError, match="array must have shape"):
        flip(src, direction="horizontal")


def test_rotate_0_returns_same() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = rotate(src, degrees=0)
    assert np.array_equal(out, src)


def test_rotate_180_reverses_content() -> None:
    src = np.arange(4, dtype=np.uint8).reshape(2, 2)
    out = rotate(src, degrees=180)
    assert out[0, 0] == src[1, 1]
    assert out[1, 1] == src[0, 0]


def test_rotate_270_shape() -> None:
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = rotate(src, degrees=270)
    assert out.shape == (3, 2)


def test_flip_vertical_reverses_rows() -> None:
    src = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    out = flip(src, direction="vertical")
    assert out[0, 0] == 3
    assert out[1, 0] == 1


def test_flip_transpose_swaps_axes() -> None:
    src = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    out = flip(src, direction="transpose")
    assert out.shape == (3, 2)
    assert out[0, 1] == 4
    assert out[2, 0] == 3


def test_flip_rejects_invalid_direction() -> None:
    src = np.zeros((2, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="direction must be one of"):
        flip(src, direction="diagonal")
