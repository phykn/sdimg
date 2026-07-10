import numpy as np
import pytest

from sdimg.image import (
    adjust_brightness_contrast,
    apply_clahe,
    equalize_histogram,
    normalize_minmax,
    normalize_zscore,
)


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("function", [equalize_histogram, apply_clahe])
def test_histogram_operations_support_all_channels_and_preserve_alpha(
    channels: int,
    function: object,
) -> None:
    image = np.zeros((8, 8, channels), dtype=np.uint8)
    image[..., 0] = np.arange(64, dtype=np.uint8).reshape(8, 8)
    if channels in {2, 4}:
        image[..., -1] = 99
    out = function(image)  # type: ignore[operator]
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    if channels in {2, 4}:
        assert np.all(out[..., -1] == 99)


def test_normalize_minmax_operates_per_channel() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = [[0, 1], [2, 3]]
    image[..., 1] = [[100, 110], [120, 130]]
    image[..., 2] = 7
    out = normalize_minmax(image)
    assert out[..., 0].min() == 0 and out[..., 0].max() == 255
    assert out[..., 1].min() == 0 and out[..., 1].max() == 255
    assert np.all(out[..., 2] == 128)


def test_normalize_zscore_constant_channels_become_midgray() -> None:
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 0], image[..., 1], image[..., 2] = 5, 100, 250
    assert np.all(normalize_zscore(image) == 128)


def test_brightness_is_independent_of_contrast_multiplier() -> None:
    image = np.full((2, 2), 127, dtype=np.uint8)
    out = adjust_brightness_contrast(image, brightness=0.1, contrast=1.0)
    assert np.all(out == 152)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.1, 1.1])
def test_adjustment_rejects_non_finite_or_out_of_range_values(value: float) -> None:
    with pytest.raises(ValueError):
        adjust_brightness_contrast(
            np.zeros((2, 2), dtype=np.uint8),
            brightness=value,
        )


@pytest.mark.parametrize("std_range", [0.0, -1.0, float("nan")])
def test_zscore_rejects_invalid_std_range(std_range: float) -> None:
    with pytest.raises(ValueError):
        normalize_zscore(np.zeros((2, 2), dtype=np.uint8), std_range=std_range)
