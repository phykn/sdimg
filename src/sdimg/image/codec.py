import base64
import io

import numpy as np
from PIL import Image

from .pillow import prepare_pillow_array


def encode_image(
    image: np.ndarray,
    *,
    method: int = 0,
    quality: int = 0,
) -> str:
    method = _validate_int_range(method, "method", 0, 6)
    quality = _validate_int_range(quality, "quality", 0, 100)
    pillow_array, channels = prepare_pillow_array(image)
    prefix = {1: b"L", 3: b"R", 4: b"A"}[channels]

    try:
        buffer = io.BytesIO()
        Image.fromarray(pillow_array).save(
            buffer,
            format="WebP",
            lossless=True,
            method=method,
            quality=quality,
            exact=True,
        )
    except Exception as exc:
        raise RuntimeError(f"encode_image failed: {exc}") from exc
    return base64.b64encode(prefix + buffer.getvalue()).decode("ascii")


def decode_image(encoded: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise TypeError("encoded must be a str.")
    try:
        payload = base64.b64decode(encoded, validate=True)
        prefix = payload[:1]
        if prefix not in {b"L", b"R", b"A"}:
            raise ValueError("invalid payload prefix")
        with Image.open(io.BytesIO(payload[1:])) as image:
            mode = {b"L": "L", b"R": "RGB", b"A": "RGBA"}[prefix]
            return np.array(image.convert(mode), dtype=np.uint8)
    except Exception as exc:
        raise ValueError(f"decode_image failed: {exc}") from exc


def _validate_int_range(value: object, name: str, minimum: int, maximum: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value
