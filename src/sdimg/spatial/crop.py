import numpy as np

from ..core.types import BBox
from ..core.validation import validate_bbox, validate_source


def crop(array: np.ndarray, bbox: BBox) -> np.ndarray:
    array = validate_source(array)
    xmin, ymin, xmax, ymax = validate_bbox(bbox, shape=array.shape[:2])
    return array[ymin:ymax, xmin:xmax].copy(order="C")
