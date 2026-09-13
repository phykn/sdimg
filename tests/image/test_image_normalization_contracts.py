import numpy as np
import pytest

from sdimg.image import normalize_minmax, normalize_zscore


@pytest.mark.parametrize("function", [normalize_minmax, normalize_zscore])
@pytest.mark.parametrize("scale", [1e308, 1e-310])
def test_normalization_is_invariant_to_extreme_scale(function, scale) -> None:
    image = np.array([[-1.0, 0.0, 1.0]])
    expected = function(image)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = function(image * scale)
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("function", [normalize_minmax, normalize_zscore])
def test_normalization_of_large_constant_is_midgray(function) -> None:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = function(np.full((2, 3), 1e308))
    np.testing.assert_array_equal(out, np.full((2, 3), 128, dtype=np.uint8))


@pytest.mark.parametrize("std_range", [1e308, 1e-310])
def test_zscore_supports_extreme_positive_std_range(std_range) -> None:
    image = np.array([[-1.0, 0.0, 1.0]])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = normalize_zscore(image, std_range=std_range)
    expected = [[128, 128, 128]] if std_range > 1 else [[0, 128, 255]]
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("function", [normalize_minmax, normalize_zscore])
@pytest.mark.parametrize("channels", [None, 1, 2, 3, 4])
def test_normalization_preserves_channel_shape_alpha_and_input(function, channels) -> None:
    image = np.array([[-1e308, 0, 1e308], [1e308, 0, -1e308]])
    if channels is not None:
        image = np.repeat(image[..., None], channels, axis=2)
        if channels in {2, 4}:
            image[..., -1] = [[0, 51, 102], [153, 204, 255]]
    original = image.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = function(image[:, ::-1])
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(image, original)
    if channels in {2, 4}:
        np.testing.assert_array_equal(out[..., -1], image[:, ::-1, -1])


@pytest.mark.parametrize("function", [normalize_minmax, normalize_zscore])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_normalization_rejects_nonfinite_visual_values(function, value) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(np.array([[0.0, value]]))


@pytest.mark.parametrize("function", [normalize_minmax, normalize_zscore])
def test_normalization_preserves_contrast_between_adjacent_large_floats(function) -> None:
    high = np.float64(1e308)
    low = np.nextafter(high, 0)
    image = np.array([[low, high]])
    expected = function(np.array([[0.0, 1.0]]))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        out = function(image)
    np.testing.assert_array_equal(out, expected)
