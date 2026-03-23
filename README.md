# sdimg

A small, function-based image and mask processing library built around `numpy.ndarray`.

No large abstractions — just a set of composable functions for preprocessing, spatial transforms, mask cleanup, and fusion workflows.

## Scope

| Module      | Functions                                                                                                                                                                                                                                                                                     |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Image**   | `clahe_norm`, `hist_norm`, `zscore_norm`, `minmax_norm`, `adjust_brightness_contrast`, `gaussian_blur`, `median_blur`, `sharpen`, `denoise`, `destripe`, `is_image`, `to_gray`, `to_rgb`, `to_uint8`                                                                                          |
| **Mask**    | `morphology` (open / close / erode / dilate), `fill_holes`, `pick_largest`, `concave_hull`, `convex_hull`, `extract_edge`, `distance_transform`, `to_roi_box`, `to_mask`, `is_mask`, `get_box_from_mask`, `get_box_from_coords`, `get_coords`, `get_centroid`, `get_roi_size`, `get_box_size` |
| **Spatial** | `resize`, `resize_keep_ratio`, `rotate`, `flip`, `pad_to_square`, `crop`, `split`, `merge`                                                                                                                                                                                                    |
| **Fusion**  | `otsu_threshold`, `grabcut`                                                                                                                                                                                                                                                                   |

## Install

```bash
pip install .
```

For destripe support (requires torch):

```bash
pip install .[destripe]
```

## Data Rules

- Inputs must be `numpy.ndarray`
- Images are returned as `np.uint8`
- Masks are returned as binary `np.uint8` with values in `{0, 1}`
- Bounding boxes follow `(wmin, hmin, wmax, hmax)` format
- Empty-mask contracts:
  `to_roi_box`, `get_box_from_mask`, `get_box_from_coords`, `get_centroid` return `None`

## Exception Rules
- **TypeError**: Raised when the input array is not a valid `numpy.ndarray`.
- **ValueError**: Raised when the input array has an invalid shape, dimension (e.g. 4D mask), or out-of-bounds parameters.
- **RuntimeError**: Deeply wrapped C-level crashes from `cv2` or `skimage` functions or external torch failures.

## Requirements

- Python 3.12+
- numpy, opencv-python-headless, scikit-image, concave-hull
- torch (optional, for `destripe`)

> **Note on `destripe`**: The `destripe` function requires `torch` to run. If `torch` is not installed, calling `destripe` will raise an `ImportError`. It automatically falls back to CPU if no NVIDIA CUDA device is found, which can be computationally intensive for large images.

## Versioning Policy

- We follow **Semantic Versioning (SemVer)**. 
- The recent `v0.1.0` to `v0.2.0` codebase refactoring separated input validation types (`TypeError` vs `ValueError`), improved in-place memory handling (avoiding unnecessary `astype(np.float32)` copies), and completely unified error throwing.

## Example

```python
import numpy as np
from sdimg.image import hist_norm, gaussian_blur
from sdimg.mask import morphology, to_roi_box
from sdimg.fusion import grabcut

image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 40:88] = 1

image = hist_norm(image)
image = gaussian_blur(image, (5, 5), 1.2)
mask = morphology(mask, "open", (3, 3), 1)

result = to_roi_box(mask)
if result is not None:
    refined = grabcut(image=image, roi=result["roi"], box=result["box"])
```

## Notebook

`test.ipynb` — visual check of the main functions using `asset/sample_image.png` and `asset/sample_mask.png`.
