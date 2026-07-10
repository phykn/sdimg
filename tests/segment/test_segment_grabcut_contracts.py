import importlib

import cv2
import numpy as np
import pytest

from sdimg.segment import refine_grabcut


def test_refine_grabcut_passes_actual_rgb_pixels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    image = np.zeros((6, 7, 3), dtype=np.uint8)
    image[..., 0] = 17
    image[..., 1] = 43
    image[..., 2] = 91
    roi = np.zeros((4, 5), dtype=np.uint8)
    roi[1:3, 1:4] = 1
    seen: dict[str, np.ndarray] = {}

    def fake_grabcut(
        *,
        img: np.ndarray,
        mask: np.ndarray,
        rect: object,
        bgdModel: np.ndarray,
        fgdModel: np.ndarray,
        iterCount: int,
        mode: int,
    ) -> None:
        seen["img"] = img.copy()
        mask[roi == 1] = cv2.GC_FGD

    monkeypatch.setattr(module.cv2, "grabCut", fake_grabcut)
    out = refine_grabcut(image, roi, (1, 1, 6, 5), iterations=1, margin=0)
    assert np.array_equal(seen["img"], image[1:5, 1:6])
    assert np.array_equal(out, roi)


def test_refine_grabcut_empty_roi_returns_binary_copy() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    roi = np.zeros((2, 3), dtype=np.uint8)
    out = refine_grabcut(image, roi, (1, 1, 4, 3), margin=0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, roi)
    assert not np.shares_memory(out, roi)


def test_refine_grabcut_full_foreground_without_margin_returns_original() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    roi = np.ones((4, 5), dtype=np.uint8)
    out = refine_grabcut(image, roi, (0, 0, 5, 4), margin=0)
    assert np.array_equal(out, roi)


def test_refine_grabcut_rejects_bbox_roi_mismatch() -> None:
    with pytest.raises(ValueError, match="roi_mask"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            (0, 0, 2, 2),
        )


def test_refine_grabcut_rejects_non_binary_roi() -> None:
    with pytest.raises(ValueError, match="binary values"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.full((3, 3), 2, dtype=np.uint8),
            (0, 0, 3, 3),
        )


def test_refine_grabcut_area_tolerance_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    roi = np.zeros((5, 5), dtype=np.uint8)
    roi[1:4, 1:4] = 1

    def erase_foreground(*, mask: np.ndarray, **kwargs: object) -> None:
        mask[:] = cv2.GC_BGD

    monkeypatch.setattr(module.cv2, "grabCut", erase_foreground)
    out = refine_grabcut(
        np.zeros((5, 5, 3), dtype=np.uint8),
        roi,
        (0, 0, 5, 5),
        margin=0,
        area_tolerance=0.0,
    )
    assert np.array_equal(out, roi)


def test_refine_grabcut_returns_binary_roi_shape() -> None:
    image = np.random.default_rng(0).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    roi = np.zeros((10, 10), dtype=np.uint8)
    roi[3:7, 3:7] = 1
    out = refine_grabcut(image, roi, (5, 5, 15, 15), iterations=1)
    assert out.dtype == np.uint8
    assert out.shape == roi.shape
    assert set(np.unique(out).tolist()) <= {0, 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"iterations": 0},
        {"margin": -1},
        {"margin": True},
        {"area_tolerance": -0.1},
        {"area_tolerance": float("nan")},
    ],
)
def test_refine_grabcut_rejects_invalid_parameters(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
            (0, 0, 3, 3),
            **kwargs,
        )


def test_refine_grabcut_wraps_opencv_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module.cv2, "grabCut", fail)
    roi = np.zeros((3, 3), dtype=np.uint8)
    roi[1, 1] = 1
    with pytest.raises(RuntimeError, match="refine_grabcut failed"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            roi,
            (0, 0, 3, 3),
            margin=0,
        )
