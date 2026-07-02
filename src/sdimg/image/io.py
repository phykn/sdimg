from pathlib import Path

import numpy as np
from PIL import Image

from .._core.validate import ensure_ndarray


def imread(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            return np.array(image.convert("RGB"))
    except Exception as exc:
        raise RuntimeError(f"imread failed: {exc}") from exc


def imwrite(path: str | Path, image: np.ndarray, **kwargs: object) -> None:
    image = ensure_ndarray(image, name="image")
    image_for_pillow = _prepare_write_image(image)

    try:
        Image.fromarray(image_for_pillow).convert("RGB").save(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"imwrite failed: {exc}") from exc


def _prepare_write_image(image: np.ndarray) -> np.ndarray:
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(
            "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    channels = image.shape[2]
    if channels == 1:
        return image[..., 0]
    if channels in {3, 4}:
        return image
    raise ValueError(
        "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
    )
