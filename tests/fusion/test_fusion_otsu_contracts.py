import numpy as np
import pytest

from sdimg.fusion import otsu_threshold


def test_otsu_threshold_returns_binary_uint8() -> None:
    src = np.array([[0, 10, 200], [30, 50, 255]], dtype=np.uint8)
    out = otsu_threshold(src, scale=1.0)

    assert out.dtype == np.uint8
    assert out.shape == src.shape
    assert set(np.unique(out).tolist()) <= {0, 1}


def test_otsu_threshold_rejects_negative_scale() -> None:
    src = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="scale must be greater than or equal to 0"):
        otsu_threshold(src, scale=-0.1)


def test_otsu_threshold_wraps_cv2_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import sdimg.fusion.otsu as otsu_module

    def _boom(*args: object, **kwargs: object) -> tuple[float, np.ndarray]:
        raise RuntimeError("boom")

    monkeypatch.setattr(otsu_module.cv2, "threshold", _boom)

    with pytest.raises(RuntimeError, match="otsu_threshold failed"):
        otsu_threshold(np.zeros((4, 4), dtype=np.uint8))


def test_otsu_threshold_binarizes_bimodal_image() -> None:
    src = np.zeros((10, 10), dtype=np.uint8)
    src[:5, :] = 200
    out = otsu_threshold(src, scale=1.0)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}
    assert np.all(out[:5, :] == 1)
    assert np.all(out[5:, :] == 0)


def test_otsu_threshold_scale_zero_makes_nonzero_foreground() -> None:
    src = np.array([[0, 1, 100], [0, 50, 255]], dtype=np.uint8)
    out = otsu_threshold(src, scale=0.0)
    assert out[0, 0] == 0
    assert out[0, 1] == 1
    assert out[1, 2] == 1


def test_otsu_threshold_uniform_image() -> None:
    src = np.full((5, 5), 128, dtype=np.uint8)
    out = otsu_threshold(src)
    assert out.dtype == np.uint8
    assert set(np.unique(out).tolist()) <= {0, 1}
