import cv2
import numpy as np

from ..image.helper import to_gray
from ..spatial.crop import crop


def _get_k(shape: tuple[int, int]) -> int:
    h, w = shape
    k = int(round(min(h, w) / 5.0))
    k = max(3, k)
    return k if k % 2 == 1 else k + 1


def _blur_mask(mask: np.ndarray) -> np.ndarray:
    k = _get_k(shape=mask.shape[:2])
    blur = cv2.GaussianBlur(
        src=mask.astype(np.float32, copy=False),
        ksize=(k, k),
        sigmaX=0.0,
        sigmaY=0.0,
    )
    return np.rint(np.clip(blur * 255.0, 0.0, 255.0)).astype(np.uint8)


def _edge(gray: np.ndarray) -> np.ndarray:
    median = cv2.medianBlur(src=gray, ksize=3)
    grad = cv2.morphologyEx(
        src=median,
        op=cv2.MORPH_GRADIENT,
        kernel=np.ones((3, 3), dtype=np.uint8),
    )
    return cv2.normalize(
        src=grad,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )


def _build_img(image: np.ndarray, roi: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    edge_map = _edge(gray=gray)
    blur = _blur_mask(mask=roi)
    return np.stack([gray, edge_map, blur], axis=2)


def _build_mask(roi: np.ndarray) -> np.ndarray:
    din = cv2.distanceTransform(roi, cv2.DIST_L2, 3, dstType=cv2.CV_32F)
    dout = cv2.distanceTransform(1 - roi, cv2.DIST_L2, 3, dstType=cv2.CV_32F)
    max_in = float(din.max())
    max_out = float(dout.max())
    th_in = min(max_in, max_in / 10.0)
    th_out = min(max_out, max_out / 10.0)

    mask = np.full(roi.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    if th_in <= 0:
        mask[roi == 1] = cv2.GC_PR_FGD
        return mask

    mask[din >= th_in] = cv2.GC_PR_FGD
    mask[din >= (2.0 * th_in)] = cv2.GC_FGD
    if th_out > 0:
        mask[dout >= (2.0 * th_out)] = cv2.GC_BGD
    return mask


def grabcut(
    image: np.ndarray,
    roi: np.ndarray,
    box: tuple[int, int, int, int],
    iter_count: int = 5,
    margin: int = 20,
) -> np.ndarray:
    assert iter_count > 0, "iter_count must be greater than 0."
    assert margin > 0, "margin must be greater than 0."
    if np.count_nonzero(roi) == 0:
        return roi

    image = crop(src=image, bbox=box)

    image = cv2.copyMakeBorder(
        src=image,
        top=margin,
        bottom=margin,
        left=margin,
        right=margin,
        borderType=cv2.BORDER_REFLECT,
    )
    roi = cv2.copyMakeBorder(
        src=roi,
        top=margin,
        bottom=margin,
        left=margin,
        right=margin,
        borderType=cv2.BORDER_CONSTANT,
        value=0,
    )

    feat = _build_img(image=image, roi=roi)
    gc_mask = _build_mask(roi=roi)

    bgd = np.zeros((1, 65), dtype=np.float64)
    fgd = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(
        img=feat,
        mask=gc_mask,
        rect=None,
        bgdModel=bgd,
        fgdModel=fgd,
        iterCount=iter_count,
        mode=cv2.GC_INIT_WITH_MASK,
    )

    out = np.isin(gc_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
    return out[margin:-margin, margin:-margin]
