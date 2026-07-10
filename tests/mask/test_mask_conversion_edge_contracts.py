import numpy as np
import pytest

from sdimg.mask import convert_to_mask, is_mask


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[False, True]], dtype=np.bool_),
        np.array([[0, 1]], dtype=np.uint8),
        np.array([[0, 255]], dtype=np.uint8),
        np.array([[0.0, 1.0]], dtype=np.float32),
    ],
)
def test_convert_to_mask_normalizes_binary_representations(mask: np.ndarray) -> None:
    out = convert_to_mask(mask)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1]]


@pytest.mark.parametrize(
    "mask",
    [
        np.array([[0, 2]], dtype=np.uint8),
        np.array([[0.0, 0.5]], dtype=np.float32),
        np.array([[0.0, np.nan]], dtype=np.float32),
        np.zeros((0, 2), dtype=np.uint8),
    ],
)
def test_convert_to_mask_rejects_invalid_values_or_shape(mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        convert_to_mask(mask)


def test_is_mask_returns_false_instead_of_raising() -> None:
    assert is_mask(np.array([[0, 255]], dtype=np.uint8)) is True
    assert is_mask(np.array([[0, 3]], dtype=np.uint8)) is False
    assert is_mask("not-mask") is False
