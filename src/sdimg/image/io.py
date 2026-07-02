from pathlib import Path

import numpy as np
from PIL import Image

from ._array import scale_to_uint8, to_pillow_uint8_array, to_rgb_uint8


def imread(path: str | Path) -> np.ndarray:
    try:
        with Image.open(path) as image:
            array = np.array(image)
            if array.dtype == np.uint8:
                return np.array(image.convert("RGB"))
            return to_rgb_uint8(scale_to_uint8(array))
    except Exception as exc:
        raise RuntimeError(f"imread failed: {exc}") from exc


def imwrite(path: str | Path, image: np.ndarray, **kwargs: object) -> None:
    image_for_pillow, _ = to_pillow_uint8_array(image)

    try:
        Image.fromarray(image_for_pillow).convert("RGB").save(path, **kwargs)
    except Exception as exc:
        raise RuntimeError(f"imwrite failed: {exc}") from exc
