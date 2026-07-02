import numpy as np
import pytest

from sdimg.mask import is_mask, to_mask


def test_to_mask_accepts_bool_and_returns_uint8_binary() -> None:
    src = np.array([[True, False], [False, True]])
    out = to_mask(src)

    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_to_mask_rejects_non_binary_values() -> None:
    src = np.array([[0, 2], [1, 1]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)


def test_is_mask_returns_true_for_valid_masks() -> None:
    assert is_mask(np.array([[0, 1], [1, 0]], dtype=np.uint8)) is True
    assert is_mask(np.array([[True, False], [False, True]])) is True
    assert is_mask(np.array([[0, 255], [255, 0]], dtype=np.uint8)) is True


def test_is_mask_returns_false_for_invalid_inputs() -> None:
    assert is_mask("not-an-array") is False
    assert is_mask(np.array([[0, 2], [1, 1]], dtype=np.uint8)) is False
    assert is_mask(np.zeros((2, 3, 4), dtype=np.uint8)) is False


def test_to_mask_accepts_all_zero_uint8() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.shape == (5, 5)
    assert int(out.sum()) == 0


def test_to_mask_accepts_all_one_uint8() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 25


def test_to_mask_accepts_zero_and_255_uint8() -> None:
    src = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [1, 0]]


def test_to_mask_accepts_all_255_uint8() -> None:
    src = np.full((3, 3), 255, dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 9


def test_to_mask_rejects_mixed_0_1_255() -> None:
    src = np.array([[0, 1, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)


def test_to_mask_accepts_float_dtype_with_binary_values() -> None:
    # Current behavior: floats with {0.0, 1.0} are accepted because set subset
    # check succeeds. After refactor the behavior must be preserved.
    src = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [1, 0]]


def test_to_mask_rejects_float_with_non_binary_values() -> None:
    src = np.array([[0.5]], dtype=np.float32)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)


def test_to_mask_rejects_float_probability_map() -> None:
    src = np.array([[0.2, 0.8], [0.3, 0.9]], dtype=np.float32)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)
