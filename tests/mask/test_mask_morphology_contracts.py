import numpy as np
import pytest

from sdimg.mask import morphology


def test_morphology_returns_binary_uint8() -> None:
    src = np.zeros((7, 7), dtype=np.uint8)
    src[2:5, 2:5] = 1
    out = morphology(src, op="erode", ksize=(3, 3), iterations=1)

    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_morphology_rejects_non_binary_mask() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        morphology(src, op="open")


def test_morphology_dilate_expands_mask() -> None:
    src = np.zeros((9, 9), dtype=np.uint8)
    src[4, 4] = 1
    out = morphology(src, op="dilate", ksize=(3, 3), iterations=1)
    assert np.count_nonzero(out) > 1


def test_morphology_open_removes_small_noise() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[3:7, 3:7] = 1
    src[0, 0] = 1
    out = morphology(src, op="open", ksize=(3, 3))
    assert out[0, 0] == 0
    assert np.count_nonzero(out[3:7, 3:7]) > 0


def test_morphology_close_fills_small_gap() -> None:
    src = np.zeros((9, 9), dtype=np.uint8)
    src[2:7, 2:7] = 1
    src[4, 4] = 0
    out = morphology(src, op="close", ksize=(3, 3))
    assert out[4, 4] == 1


def test_morphology_empty_mask_returns_empty() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    for op in ("erode", "dilate", "open", "close"):
        out = morphology(src, op=op)
        assert np.count_nonzero(out) == 0


def test_morphology_rejects_invalid_op() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    src[2, 2] = 1
    with pytest.raises(ValueError, match="op must be one of"):
        morphology(src, op="invalid")
