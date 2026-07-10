import numpy as np
import pytest

from sdimg.core.validation import (
    validate_bbox,
    validate_finite,
    validate_image,
    validate_positive_int,
    validate_source,
)


@pytest.mark.parametrize(
    "array",
    [
        np.zeros((0, 2), dtype=np.uint8),
        np.zeros((2, 0), dtype=np.uint8),
        np.zeros((2, 2), dtype=object),
        np.zeros((2, 2), dtype=np.complex64),
    ],
)
def test_validate_image_rejects_empty_or_non_real_images(array: np.ndarray) -> None:
    with pytest.raises(ValueError):
        validate_image(array)


def test_validate_source_accepts_real_2d_and_3d_arrays() -> None:
    assert validate_source(np.zeros((2, 3), dtype=np.float32)).shape == (2, 3)
    assert validate_source(np.zeros((2, 3, 4), dtype=np.uint8)).shape == (2, 3, 4)


def test_validate_source_rejects_empty_channel_axis() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_source(np.zeros((2, 2, 0), dtype=np.uint8))


@pytest.mark.parametrize(
    "bbox,exception",
    [
        ((True, 0, 2, 2), TypeError),
        ((0.0, 0, 2, 2), TypeError),
        ((2, 0, 1, 2), ValueError),
        ((0, 0, 5, 2), ValueError),
    ],
)
def test_validate_bbox_rejects_invalid_values(
    bbox: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        validate_bbox(bbox, shape=(4, 4))


@pytest.mark.parametrize(
    "value,exception",
    [(True, TypeError), (0, ValueError), (-1, ValueError), (1.5, TypeError)],
)
def test_validate_positive_int_rejects_invalid_values(
    value: object,
    exception: type[Exception],
) -> None:
    with pytest.raises(exception):
        validate_positive_int(value, name="size")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validate_finite_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite"):
        validate_finite(value, name="value")
