import numpy as np
import pytest

from sdimg.fusion import grabcut


def test_grabcut_rejects_invalid_iter_count_and_margin() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    roi = np.ones((5, 5), dtype=np.uint8)
    box = (0, 0, 5, 5)

    with pytest.raises(ValueError, match="iter_count must be greater than 0"):
        grabcut(image=image, roi=roi, box=box, iter_count=0)

    with pytest.raises(ValueError, match="margin must be greater than 0"):
        grabcut(image=image, roi=roi, box=box, margin=0)


def test_grabcut_returns_same_mask_for_empty_roi() -> None:
    image = np.zeros((6, 6, 3), dtype=np.uint8)
    roi = np.zeros((3, 3), dtype=np.uint8)
    box = (1, 1, 4, 4)

    out = grabcut(image=image, roi=roi, box=box)

    assert np.array_equal(out, roi)


def test_grabcut_rejects_shape_mismatch() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    roi = np.ones((4, 4), dtype=np.uint8)
    box = (0, 0, 3, 3)

    with pytest.raises(ValueError, match="does not match roi shape"):
        grabcut(image=image, roi=roi, box=box)


def test_grabcut_rejects_negative_tol() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    roi = np.ones((5, 5), dtype=np.uint8)
    box = (0, 0, 5, 5)

    with pytest.raises(ValueError, match="tol must be greater than or equal to 0"):
        grabcut(image=image, roi=roi, box=box, tol=-0.1)


def test_grabcut_rejects_non_binary_roi() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    roi = np.full((5, 5), 2, dtype=np.uint8)
    box = (0, 0, 5, 5)

    with pytest.raises(ValueError, match="binary values"):
        grabcut(image=image, roi=roi, box=box)


def test_grabcut_wraps_cv2_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib

    grabcut_module = importlib.import_module("sdimg.fusion.grabcut")

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("cv2 boom")

    monkeypatch.setattr(grabcut_module.cv2, "grabCut", _boom)

    image = np.zeros((10, 10, 3), dtype=np.uint8)
    roi = np.ones((5, 5), dtype=np.uint8)
    box = (0, 0, 5, 5)

    with pytest.raises(RuntimeError, match="grabcut failed"):
        grabcut(image=image, roi=roi, box=box)


def test_grabcut_returns_binary_mask_with_correct_shape() -> None:
    image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
    roi = np.zeros((10, 10), dtype=np.uint8)
    roi[3:7, 3:7] = 1
    box = (5, 5, 15, 15)

    out = grabcut(image=image, roi=roi, box=box, iter_count=1)
    assert out.dtype == np.uint8
    assert out.shape == (10, 10)
    assert set(np.unique(out).tolist()) <= {0, 1}
