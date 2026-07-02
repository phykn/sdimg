import numpy as np
import pytest

from sdimg.mask import is_mask, to_mask


@pytest.mark.parametrize(
    "src,expected",
    [
        (np.array([[True, False], [False, True]]), [[1, 0], [0, 1]]),
        (np.zeros((2, 2), dtype=np.uint8), [[0, 0], [0, 0]]),
        (np.ones((2, 2), dtype=np.uint8), [[1, 1], [1, 1]]),
        (np.array([[0, 255], [255, 0]], dtype=np.uint8), [[0, 1], [1, 0]]),
        (np.full((2, 2), 255, dtype=np.uint8), [[1, 1], [1, 1]]),
        (np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32), [[0, 1], [1, 0]]),
    ],
)
def test_to_mask_accepts_binary_representations(
    src: np.ndarray,
    expected: list[list[int]],
) -> None:
    out = to_mask(src)

    assert out.dtype == np.uint8
    assert out.tolist() == expected


def test_is_mask_returns_true_for_valid_masks() -> None:
    assert is_mask(np.array([[0, 1], [1, 0]], dtype=np.uint8)) is True
    assert is_mask(np.array([[True, False], [False, True]])) is True
    assert is_mask(np.array([[0, 255], [255, 0]], dtype=np.uint8)) is True


def test_is_mask_returns_false_for_invalid_inputs() -> None:
    assert is_mask("not-an-array") is False
    assert is_mask(np.array([[0, 2], [1, 1]], dtype=np.uint8)) is False
    assert is_mask(np.zeros((2, 3, 4), dtype=np.uint8)) is False


@pytest.mark.parametrize(
    "src",
    [
        np.array([[0, 2], [1, 1]], dtype=np.uint8),
        np.array([[0, 1, 255]], dtype=np.uint8),
        np.array([[0.5]], dtype=np.float32),
        np.array([[0.2, 0.8], [0.3, 0.9]], dtype=np.float32),
    ],
)
def test_to_mask_rejects_non_binary_values(src: np.ndarray) -> None:
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)
