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
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1, 1] = 1
    mask[2:4, 2:5] = 1
    mask[4, 5] = 1
    roi = mask[1:5, 1:6]
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
        mask[:] = cv2.GC_BGD
        mask[roi == 1] = cv2.GC_FGD

    monkeypatch.setattr(module.cv2, "grabCut", fake_grabcut)
    out = refine_grabcut(image, mask, iterations=1, margin=0)
    assert np.array_equal(seen["img"], image[1:5, 1:6])
    assert np.array_equal(out, mask)


def test_refine_grabcut_restores_edge_roi_to_full_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    image = np.zeros((6, 7, 3), dtype=np.uint8)
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[0, 0] = 1
    mask[0:2, 1:3] = 1
    roi = mask[0:3, 0:4]
    seen: dict[str, np.ndarray] = {}

    def fake_grabcut(
        *, img: np.ndarray, mask: np.ndarray, **kwargs: object
    ) -> None:
        seen["img"] = img.copy()
        mask[:] = cv2.GC_BGD
        mask[roi == 1] = cv2.GC_FGD

    monkeypatch.setattr(module.cv2, "grabCut", fake_grabcut)
    out = refine_grabcut(image, mask, iterations=1, margin=1, area_tolerance=0.0)
    assert out.shape == image.shape[:2]
    assert np.array_equal(seen["img"], image[0:3, 0:4])
    assert np.array_equal(out, mask)


def test_refine_grabcut_can_expand_beyond_initial_bbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    image = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    mask = np.zeros((8, 9), dtype=np.uint8)
    mask[3:5, 4:6] = 1
    seen: dict[str, np.ndarray] = {}

    def fake_grabcut(
        *, img: np.ndarray, mask: np.ndarray, **kwargs: object
    ) -> None:
        seen["img"] = img.copy()
        mask[:] = cv2.GC_BGD
        mask[1:3, 1:3] = cv2.GC_FGD
        mask[1, 3] = cv2.GC_FGD

    monkeypatch.setattr(module.cv2, "grabCut", fake_grabcut)
    out = refine_grabcut(image, mask, iterations=1, margin=1, area_tolerance=0.25)
    assert np.array_equal(seen["img"], image[2:6, 3:7])
    assert out[3, 6] == 1
    assert np.count_nonzero(out) == 5


def test_refine_grabcut_can_expand_at_image_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    image = np.zeros((5, 6, 3), dtype=np.uint8)
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[0:2, 0:2] = 1

    def fake_grabcut(*, mask: np.ndarray, **kwargs: object) -> None:
        mask[:] = cv2.GC_BGD
        mask[0:2, 0:2] = cv2.GC_FGD
        mask[2, 2] = cv2.GC_FGD

    monkeypatch.setattr(module.cv2, "grabCut", fake_grabcut)
    out = refine_grabcut(image, mask, iterations=1, margin=1, area_tolerance=0.25)
    assert out[2, 2] == 1
    assert np.count_nonzero(out) == 5


def test_refine_grabcut_empty_mask_returns_binary_copy() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.zeros((4, 5), dtype=np.uint8)
    out = refine_grabcut(image, mask, margin=0)
    assert out.dtype == np.uint8
    assert np.array_equal(out, mask)
    assert not np.shares_memory(out, mask)


def test_refine_grabcut_full_foreground_without_margin_returns_original() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = np.ones((4, 5), dtype=np.uint8)
    out = refine_grabcut(image, mask, margin=0)
    assert np.array_equal(out, mask)


def test_refine_grabcut_rejects_image_mask_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.ones((3, 3), dtype=np.uint8),
        )


def test_refine_grabcut_rejects_non_binary_mask() -> None:
    with pytest.raises(ValueError, match="binary values"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            np.full((5, 5), 2, dtype=np.uint8),
        )


def test_refine_grabcut_area_tolerance_returns_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    mask[1, 1] = 0

    def erase_foreground(*, mask: np.ndarray, **kwargs: object) -> None:
        mask[:] = cv2.GC_BGD

    monkeypatch.setattr(module.cv2, "grabCut", erase_foreground)
    out = refine_grabcut(
        np.zeros((5, 5, 3), dtype=np.uint8),
        mask,
        margin=0,
        area_tolerance=0.0,
    )
    assert np.array_equal(out, mask)


def test_refine_grabcut_returns_binary_image_shape() -> None:
    image = np.random.default_rng(0).integers(0, 256, (20, 20, 3), dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[8:12, 8:12] = 1
    out = refine_grabcut(image, mask, iterations=1)
    assert out.dtype == np.uint8
    assert out.shape == mask.shape
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
            np.ones((5, 5), dtype=np.uint8),
            **kwargs,
        )


def test_refine_grabcut_wraps_opencv_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("sdimg.segment.grabcut")

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(module.cv2, "grabCut", fail)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[1, 1] = 1
    mask[3, 3] = 1
    with pytest.raises(RuntimeError, match="refine_grabcut failed"):
        refine_grabcut(
            np.zeros((5, 5, 3), dtype=np.uint8),
            mask,
            margin=0,
        )
