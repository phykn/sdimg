import numpy as np


def pad_1px(mask: np.ndarray) -> np.ndarray:
    return np.pad(
        mask,
        ((1, 1), (1, 1)),
        mode="constant",
        constant_values=0,
    )


def unpad_1px(mask: np.ndarray) -> np.ndarray:
    return mask[1:-1, 1:-1]
