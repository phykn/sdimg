import importlib

import numpy as np
import pytest

from sdimg.segment import threshold_otsu


def test_threshold_otsu_binarizes_bimodal_image() -> None:
    image = np.zeros((10, 10), dtype=np.uint8)
    image[:, 5:] = 200
    out = threshold_otsu(image)
    assert out.dtype == np.uint8
    assert np.all(out[:, :5] == 0)
    assert np.all(out[:, 5:] == 1)


def test_threshold_otsu_scale_zero_marks_nonzero_pixels() -> None:
    image = np.array([[0, 1], [2, 0]], dtype=np.uint8)
    assert np.array_equal(threshold_otsu(image, scale=0.0), image > 0)


def test_threshold_otsu_uniform_image_returns_binary_mask() -> None:
    out = threshold_otsu(np.full((6, 6), 50, dtype=np.uint8))
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}


@pytest.mark.parametrize("scale", [-1.0, float("nan"), float("inf")])
def test_threshold_otsu_rejects_invalid_scale(scale: float) -> None:
    with pytest.raises(ValueError):
        threshold_otsu(np.zeros((2, 2), dtype=np.uint8), scale=scale)


def test_threshold_otsu_wraps_opencv_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    module = importlib.import_module("sdimg.segment.threshold")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module.cv2, "threshold", fail)
    with pytest.raises(RuntimeError, match="threshold_otsu failed"):
        threshold_otsu(np.zeros((2, 2), dtype=np.uint8))
