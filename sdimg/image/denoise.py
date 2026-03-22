import cv2
import numpy as np

from .helper import to_gray, to_rgb


def denoise(
    image: np.ndarray,
    h: float = 3.0,
    hColor: float = 3.0,
    templateWindowSize: int = 7,
    searchWindowSize: int = 21,
) -> np.ndarray:
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.fastNlMeansDenoisingColored(
            image,
            h=h,
            hColor=hColor,
            templateWindowSize=templateWindowSize,
            searchWindowSize=searchWindowSize,
        )

    return cv2.fastNlMeansDenoising(
        image,
        h=h,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize,
    )


def destripe(
    image: np.ndarray,
    mu1: float = 0.33,
    mu2: float = 0.003,
    iterations: int = 500,
    n_tiles: int = 1,
    verbose: bool = False,
) -> np.ndarray:
    import torch

    from .remove_stripe import UniversalStripeRemover

    is_gray = image.ndim == 2
    gray = to_gray(image)
    x = gray.astype(np.float32) / 255.0
    x_torch = torch.from_numpy(x).unsqueeze(0)

    remover = UniversalStripeRemover(mu1=mu1, mu2=mu2)

    if n_tiles > 1:
        res = remover.process_tiled(
            x_torch,
            n=n_tiles,
            iterations=iterations,
            verbose=verbose,
        )
    else:
        res = remover.process(
            x_torch,
            iterations=iterations,
            verbose=verbose,
        )

    res_np = res.numpy().squeeze(0)
    res_np = (res_np * 255.0).clip(0, 255).astype(np.uint8)

    if is_gray:
        return res_np
    return to_rgb(res_np)
