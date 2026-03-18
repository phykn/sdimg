# sdimg

`sdimg` is a small function-based image processing library built around `numpy.ndarray`.
It is built for practical work with image arrays and binary masks, especially when the goal is to keep the workflow simple and direct.
The library covers common operations such as preprocessing, geometric transforms, mask cleanup, and synchronized handling of image-mask pairs.
Instead of introducing large abstractions, it exposes a small set of functions that can be used together as needed.
This makes it easier to use in scripts, experiments, and lightweight processing pipelines.
In short, `sdimg` is meant to be a compact toolkit for common image and mask workflows.

## Scope

- Image: normalization, brightness and contrast adjustment, blur, denoising, sharpening
- Mask: morphology, hole filling, largest-component selection, hull generation, edge extraction, distance transform, and helper utilities
- Spatial: resize, rotate, flip, pad, crop, split, and merge
- Fusion: Otsu thresholding, mask refinement, and synchronized image-mask transforms

## Requirements

- Python 3.12+
- numpy
- opencv-python-headless
- scikit-image
- concave-hull

## Install

```bash
pip install -r requirements.txt
```

or

```bash
pip install .
```

## Build Wheel

```bash
python -m pip install build
python -m build
```

Built files are created in `dist/`.

## Data Rules

- Inputs must be `numpy.ndarray`
- Images are returned as `np.uint8`
- Masks are returned as binary `np.uint8` arrays with values in `{0, 1}`

## Example

```python
import numpy as np
import sdimg

image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
mask = np.zeros((128, 128), dtype=np.uint8)
mask[32:96, 40:88] = 1

image = sdimg.hist_norm(image)
image = sdimg.gaussian_blur(image, (5, 5), 1.2)
mask = sdimg.morphology(mask, "open", (3, 3), 1)

resized_image, resized_mask = sdimg.sync_resize(image, mask, height=256)
```

## Notebook

`test.ipynb` is provided for visual checking of the main functions.
It uses `asset/sample_image.png` and `asset/sample_mask.png` and is intended as a simple manual test notebook.
