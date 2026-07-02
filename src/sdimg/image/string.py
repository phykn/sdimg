import base64
import io

import numpy as np
from PIL import Image

from .._core.validate import ensure_ndarray


def encode(image: np.ndarray, *, method: int = 0, quality: int = 0) -> str:
    image = ensure_ndarray(image, name="image")
    payload_prefix, image_for_pillow = _prepare_encode_image(image)

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


def _prepare_encode_image(image: np.ndarray) -> tuple[bytes, np.ndarray]:
    if image.dtype != np.uint8:
        raise ValueError("image must have dtype uint8.")
    if image.ndim == 2:
        return b"L", image
    if image.ndim != 3:
        raise ValueError(
            "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
        )

    channels = image.shape[2]
    if channels == 1:
        return b"L", image[..., 0]
    if channels == 3:
        return b"R", image
    if channels == 4:
        return b"A", image
    raise ValueError(
        "image must have shape (H, W), (H, W, 1), (H, W, 3), or (H, W, 4)."
    )
