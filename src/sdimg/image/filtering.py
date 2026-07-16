import cv2
import numpy as np

from ..core.validation import validate_finite, validate_image, validate_positive_int
from .channels import prepare_visual_alpha, restore_visual_alpha


def apply_gaussian_blur(
    image: np.ndarray,
    kernel_size: tuple[int, int],
    sigma_x: float,
    sigma_y: float = 0.0,
    border_type: int = cv2.BORDER_DEFAULT,
) -> np.ndarray:
    image = validate_image(image)
    _validate_kernel_size(kernel_size)
    sigma_x = validate_finite(sigma_x, "sigma_x")
    sigma_y = validate_finite(sigma_y, "sigma_y")
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError("sigma_x and sigma_y must be greater than or equal to 0.")
    if not isinstance(border_type, int) or isinstance(border_type, bool):
        raise TypeError("border_type must be an int.")

    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    try:
        result = cv2.GaussianBlur(
            visual,
            kernel_size,
            sigma_x,
            sigmaY=sigma_y,
            borderType=border_type,
        )
    except Exception as exc:
        raise RuntimeError(f"apply_gaussian_blur failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


def apply_median_blur(image: np.ndarray, kernel_size: int) -> np.ndarray:
    image = validate_image(image)
    kernel_size = validate_positive_int(kernel_size, "kernel_size")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    try:
        result = cv2.medianBlur(visual, kernel_size)
    except Exception as exc:
        raise RuntimeError(f"apply_median_blur failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


def denoise(
    image: np.ndarray,
    strength: float = 3.0,
    color_strength: float = 3.0,
    template_size: int = 7,
    search_size: int = 21,
) -> np.ndarray:
    image = validate_image(image)
    strength = validate_finite(strength, "strength")
    color_strength = validate_finite(color_strength, "color_strength")
    if strength < 0 or color_strength < 0:
        raise ValueError("strength values must be greater than or equal to 0.")
    template_size = _validate_odd_size(template_size, "template_size")
    search_size = _validate_odd_size(search_size, "search_size")
    if template_size > search_size:
        raise ValueError("template_size must not exceed search_size.")

    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    if strength == 0 and color_strength == 0:
        return restore_visual_alpha(visual, alpha, ndim, channels)

    try:
        if visual.ndim == 3:
            bgr = cv2.cvtColor(visual, cv2.COLOR_RGB2BGR)
            denoised = cv2.fastNlMeansDenoisingColored(
                bgr,
                h=strength,
                hColor=color_strength,
                templateWindowSize=template_size,
                searchWindowSize=search_size,
            )
            result = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        else:
            result = cv2.fastNlMeansDenoising(
                visual,
                h=strength,
                templateWindowSize=template_size,
                searchWindowSize=search_size,
            )
    except Exception as exc:
        raise RuntimeError(f"denoise failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


def sharpen(image: np.ndarray, amount: float = 1.0) -> np.ndarray:
    image = validate_image(image)
    amount = validate_finite(amount, "amount")
    if amount < 0:
        raise ValueError("amount must be greater than or equal to 0.")

    visual, alpha, ndim, channels = prepare_visual_alpha(
        image,
        convert_visual=True,
    )
    try:
        blurred = cv2.GaussianBlur(visual, (0, 0), 1.0)
        result = cv2.addWeighted(visual, 1.0 + amount, blurred, -amount, 0.0)
    except Exception as exc:
        raise RuntimeError(f"sharpen failed: {exc}") from exc
    return restore_visual_alpha(result, alpha, ndim, channels)


def _validate_kernel_size(kernel_size: object) -> tuple[int, int]:
    if not isinstance(kernel_size, tuple) or len(kernel_size) != 2:
        raise TypeError("kernel_size must be a tuple of two ints.")
    width = validate_positive_int(kernel_size[0], "kernel_size[0]")
    height = validate_positive_int(kernel_size[1], "kernel_size[1]")
    if width % 2 == 0 or height % 2 == 0:
        raise ValueError("kernel_size values must be odd.")
    return (width, height)


def _validate_odd_size(value: object, name: str) -> int:
    result = validate_positive_int(value, name)
    if result % 2 == 0:
        raise ValueError(f"{name} must be odd.")
    return result
