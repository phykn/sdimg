import importlib

import numpy as np
import pytest

from sdimg.image import (
    apply_gaussian_blur,
    apply_median_blur,
    denoise,
    sharpen,
)


def _rgba_float() -> np.ndarray:
    image = np.zeros((9, 9, 4), dtype=np.float32)
    image[4, 4, :3] = 255.0
    image[..., 3] = np.arange(81, dtype=np.float32).reshape(9, 9)
    return image


def test_gaussian_blur_converts_float_and_preserves_alpha() -> None:
    image = _rgba_float()
    out = apply_gaussian_blur(image, kernel_size=(3, 3), sigma_x=1.0)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out[..., 3], image[..., 3].astype(np.uint8))
    assert out[4, 4, 0] < 255


def test_median_blur_preserves_single_alpha_shape() -> None:
    image = _rgba_float()[..., :2]
    out = apply_median_blur(image, kernel_size=3)
    assert out.shape == image.shape
    assert np.array_equal(out[..., 1], image[..., 1].astype(np.uint8))


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_denoise_preserves_channel_shape_and_alpha(channels: int) -> None:
    image = np.zeros((10, 10, channels), dtype=np.uint8)
    if channels in {2, 4}:
        image[..., -1] = np.arange(100, dtype=np.uint8).reshape(10, 10)
    out = denoise(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8
    if channels in {2, 4}:
        assert np.array_equal(out[..., -1], image[..., -1])


def test_sharpen_returns_uint8_and_preserves_alpha() -> None:
    image = _rgba_float()
    out = sharpen(image, amount=1.0)
    assert out.dtype == np.uint8
    assert out.shape == image.shape
    assert np.array_equal(out[..., 3], image[..., 3].astype(np.uint8))


@pytest.mark.parametrize(
    "call",
    [
        lambda image: apply_gaussian_blur(image, (2, 3), 1.0),
        lambda image: apply_median_blur(image, 2),
        lambda image: sharpen(image, amount=-0.1),
    ],
)
def test_filters_reject_invalid_parameters(call: object) -> None:
    with pytest.raises(ValueError):
        call(np.zeros((5, 5), dtype=np.uint8))  # type: ignore[operator]


def test_gaussian_blur_wraps_opencv_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("sdimg.image.filtering")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module.cv2, "GaussianBlur", fail)
    with pytest.raises(RuntimeError, match="apply_gaussian_blur failed"):
        apply_gaussian_blur(np.zeros((5, 5), dtype=np.uint8), (3, 3), 1.0)
