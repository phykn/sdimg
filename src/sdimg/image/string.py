import base64
import io

import numpy as np
from PIL import Image

from ._array import to_pillow_uint8_array


def encode(image: np.ndarray, *, method: int = 0, quality: int = 0) -> str:
    image_for_pillow, channels = to_pillow_uint8_array(image)
    payload_prefix = _payload_prefix(channels)

    try:
        buffer = io.BytesIO()
        Image.fromarray(image_for_pillow).save(
            buffer,
            format="WebP",
            lossless=True,
            method=method,
            quality=quality,
            exact=True,
        )
    except Exception as exc:
        raise ValueError(f"failed to serialize array: {exc}") from exc

    return base64.b64encode(payload_prefix + buffer.getvalue()).decode("utf-8")


def decode(encoded: str) -> np.ndarray:
    if not isinstance(encoded, str):
        raise TypeError("encoded must be a str.")

    try:
        payload = base64.b64decode(encoded, validate=True)
        prefix = payload[:1]
        webp_data = payload[1:]
        if prefix not in {b"L", b"R", b"A", b"C"}:
            raise ValueError("invalid payload prefix")

        with Image.open(io.BytesIO(webp_data)) as image:
            decoded = np.array(image)

        if prefix == b"L" and decoded.ndim == 3:
            return decoded[..., 0]
        if prefix == b"R" and decoded.ndim == 3 and decoded.shape[2] >= 3:
            return decoded[..., :3]
        return decoded
    except Exception as exc:
        raise ValueError(f"failed to deserialize array: {exc}") from exc


def _payload_prefix(channels: int) -> bytes:
    if channels == 1:
        return b"L"
    if channels == 3:
        return b"R"
    if channels == 4:
        return b"A"
    raise ValueError("channels must be one of 1, 3, or 4.")
