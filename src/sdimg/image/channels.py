import numpy as np

from .conversion import convert_to_uint8


def prepare_visual_alpha(
    image: np.ndarray,
    *,
    convert_visual: bool,
) -> tuple[np.ndarray, np.ndarray | None, int, int | None]:
    if image.ndim == 2:
        visual = image
        alpha = None
    else:
        channels = image.shape[2]
        if channels == 1:
            visual = image[..., 0]
            alpha = None
        elif channels == 2:
            visual = image[..., 0]
            alpha = image[..., 1]
        elif channels == 3:
            visual = image
            alpha = None
        else:
            visual = image[..., :3]
            alpha = image[..., 3]

    if convert_visual:
        visual = convert_to_uint8(visual)
    if alpha is not None:
        alpha = convert_to_uint8(alpha)
    original_channels = image.shape[2] if image.ndim == 3 else None
    return visual, alpha, image.ndim, original_channels


def restore_visual_alpha(
    visual: np.ndarray,
    alpha: np.ndarray | None,
    original_ndim: int,
    original_channels: int | None,
) -> np.ndarray:
    if original_ndim == 2:
        return np.ascontiguousarray(visual)
    if original_channels == 1:
        return np.ascontiguousarray(visual[..., None])
    if alpha is None:
        return np.ascontiguousarray(visual)
    if visual.ndim == 2:
        visual = visual[..., None]
    return np.ascontiguousarray(
        np.concatenate([visual, alpha[..., None]], axis=2),
    )
