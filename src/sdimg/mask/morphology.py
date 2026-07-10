from typing import Literal

import cv2
import numpy as np

from ..core.validation import validate_positive_int
from .conversion import convert_to_mask

MorphologyOperation = Literal["open", "close", "erode", "dilate"]


def apply_morphology(
    mask: np.ndarray,
    operation: MorphologyOperation,
    kernel_size: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    binary = convert_to_mask(mask)
    if operation not in {"open", "close", "erode", "dilate"}:
        raise ValueError(
            "operation must be one of: 'open', 'close', 'erode', 'dilate'."
        )
    kernel = np.ones(_validate_kernel_size(kernel_size), dtype=np.uint8)
    iterations = validate_positive_int(iterations, "iterations")
    if not np.any(binary):
        return binary
    try:
        if operation == "erode":
            result = cv2.erode(
                binary,
                kernel,
                iterations=iterations,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        elif operation == "dilate":
            result = cv2.dilate(
                binary,
                kernel,
                iterations=iterations,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            code = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
            result = cv2.morphologyEx(
                binary,
                code,
                kernel,
                iterations=iterations,
                borderType=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
    except Exception as exc:
        raise RuntimeError(f"apply_morphology failed: {exc}") from exc
    return (result > 0).astype(np.uint8)


def extract_boundary(
    mask: np.ndarray,
    kernel_size: tuple[int, int] = (3, 3),
) -> np.ndarray:
    binary = convert_to_mask(mask)
    kernel = np.ones(_validate_kernel_size(kernel_size), dtype=np.uint8)
    if not np.any(binary):
        return binary
    try:
        eroded = cv2.erode(
            binary,
            kernel,
            borderType=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except Exception as exc:
        raise RuntimeError(f"extract_boundary failed: {exc}") from exc
    return (binary - eroded > 0).astype(np.uint8)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    binary = convert_to_mask(mask)
    if not np.any(binary):
        return binary
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    background = ((1 - padded) * 255).astype(np.uint8)
    flood_mask = np.zeros(
        (background.shape[0] + 2, background.shape[1] + 2),
        dtype=np.uint8,
    )
    try:
        cv2.floodFill(background, flood_mask, (0, 0), 0)
    except Exception as exc:
        raise RuntimeError(f"fill_holes failed: {exc}") from exc
    holes = (background > 0).astype(np.uint8)
    filled = np.maximum(padded, holes)
    return filled[1:-1, 1:-1].astype(np.uint8)


def _validate_kernel_size(value: object) -> tuple[int, int]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError("kernel_size must be a tuple of two ints.")
    return (
        validate_positive_int(value[0], "kernel_size[0]"),
        validate_positive_int(value[1], "kernel_size[1]"),
    )
