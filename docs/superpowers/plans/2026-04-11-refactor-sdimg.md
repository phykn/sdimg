# sdimg Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve seven findings from the 2026-04-11 refactor diagnosis: architectural duplication in `grabcut`, slow `to_mask` value validation, `helper.py` catch-all split, trivial pad wrappers, `pad.py` naming collision, and undocumented RGB / channel-count contract.

**Architecture:** Pure refactor. No new functionality, no API surface change. All public re-exports from `sdimg.image`, `sdimg.mask`, `sdimg.spatial`, `sdimg.fusion` remain stable. Existing contract tests are the oracle — they must pass after each task.

**Tech Stack:** Python ≥ 3.12, `numpy`, `opencv-python`, `pytest`.

**Tasks are ordered from lowest-risk to highest-risk.** F-05/F-06 first (docs only), then small code changes (F-08+F-01, F-03), then touchy ones (F-09, F-02).

**Global testing convention:** run `PYTHONPATH=src pytest -q` after every task. Zero failures required before commit.

---

### Task 1: Document RGB contract and channel-count semantics (F-05, F-06)

**Files:**
- Modify: `README.md` (Core Contracts section)
- Modify: `CLAUDE.md` (Core Contracts section)
- Modify: `src/sdimg/_core/validate.py` (`ensure_image` docstring)

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Update `README.md` Core Contracts**

Replace the `## Core Contracts` section of `README.md` with:

```markdown
## Core Contracts

- Input arrays must be `numpy.ndarray`
- Images: shape `(H, W)` or `(H, W, C)` with `C in 1..4`
- **Color channel order: RGB.** Color images passed to `sdimg` must be in RGB order. `cv2.imread` returns BGR — callers using OpenCV I/O must convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before calling `sdimg` functions.
- **Channel-count semantics:**
  - `C == 1`: grayscale
  - `C == 2`: grayscale + alpha (alpha is ignored by `to_gray`/`to_rgb`)
  - `C == 3`: RGB
  - `C == 4`: RGBA (alpha is ignored by `to_gray`/`to_rgb`)
- Masks: shape `(H, W)`, binary values (`bool`, `{0,1}`, `{0,255}`)
- Output images are `np.uint8`
- Output masks are binary `np.uint8` in `{0, 1}`
- BBox format: `(wmin, hmin, wmax, hmax)`
- Empty-mask returns `None` for:
  - `to_roi_box`
  - `get_box_from_mask`
  - `get_box_from_coords`
  - `get_centroid`
```

- [ ] **Step 3: Update `CLAUDE.md` Core Contracts**

In `CLAUDE.md`, replace the `## Core Contracts` section with:

```markdown
## Core Contracts

- All inputs must be `numpy.ndarray`.
- **Images**: shape `(H, W)` or `(H, W, C)` with `C ∈ {1,2,3,4}`. Output dtype is `np.uint8`.
- **Color order: RGB.** Color images are assumed to be in RGB channel order. Not enforced at runtime — it is a documented contract. Callers reading with `cv2.imread` must convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` first.
- **Channel-count semantics**: C=1 grayscale; C=2 grayscale+alpha (alpha ignored); C=3 RGB; C=4 RGBA (alpha ignored).
- **Masks**: shape `(H, W)`, binary values (`bool`, `{0,1}`, `{0,255}`). Output is `np.uint8 ∈ {0, 1}`.
- **BBox**: `(wmin, hmin, wmax, hmax)` — width-first, min-inclusive, max-exclusive.
- Empty-mask functions (`to_roi_box`, `get_box_from_mask`, `get_box_from_coords`, `get_centroid`) return `None`.
```

- [ ] **Step 4: Update `ensure_image` docstring**

In `src/sdimg/_core/validate.py`, replace the `ensure_image` function with:

```python
def ensure_image(image: object, name: str = "image") -> np.ndarray:
    """Validate an image array.

    Accepts shape (H, W) or (H, W, C) with C in {1, 2, 3, 4}.

    Channel-count semantics (not runtime-enforced, documented contract):
        C == 1: grayscale
        C == 2: grayscale + alpha (alpha ignored by to_gray/to_rgb)
        C == 3: RGB — sdimg assumes RGB channel order, not BGR
        C == 4: RGBA (alpha ignored by to_gray/to_rgb)
    """
    arr = ensure_src(image, name=name)
    if arr.ndim == 2:
        return arr
    channels = arr.shape[2]
    if channels not in {1, 2, 3, 4}:
        raise value_error(f"{name} must have shape (H, W) or (H, W, C) with C in 1..4.")
    return arr
```

- [ ] **Step 5: Run the test suite — docs only, nothing should change**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add README.md CLAUDE.md src/sdimg/_core/validate.py
git commit -m "docs: document RGB channel-order contract and channel-count semantics"
```

---

### Task 2: Inline `_pad_1px` / `_unpad_1px` and delete `mask/pad.py` (F-08, F-01)

**Files:**
- Modify: `src/sdimg/mask/morphology.py`
- Modify: `src/sdimg/mask/edge.py`
- Modify: `src/sdimg/mask/distance.py`
- Delete: `src/sdimg/mask/pad.py`

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Rewrite `mask/morphology.py` to inline the pad helpers**

Replace the entire contents of `src/sdimg/mask/morphology.py` with:

```python
import cv2
import numpy as np
from typing import Literal

from .helper import to_mask

MorphologyOp = Literal["open", "close", "erode", "dilate"]


def morphology(
    mask: np.ndarray,
    op: MorphologyOp,
    ksize: tuple[int, int] = (3, 3),
    iterations: int = 1,
) -> np.ndarray:
    mask = to_mask(mask)

    ops = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": "erode",
        "dilate": "dilate",
    }
    if op not in ops:
        raise ValueError(f"op must be one of: {', '.join(repr(k) for k in ops)}.")

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)

    if op == "erode":
        result = cv2.erode(padded, kernel, iterations=iterations)
    elif op == "dilate":
        result = cv2.dilate(padded, kernel, iterations=iterations)
    else:
        result = cv2.morphologyEx(
            padded,
            ops[op],
            kernel,
            iterations=iterations,
        )

    result = result[1:-1, 1:-1]
    return (result > 0).astype(np.uint8)
```

- [ ] **Step 3: Rewrite `mask/edge.py` to inline the pad helpers**

Replace the entire contents of `src/sdimg/mask/edge.py` with:

```python
import cv2
import numpy as np

from .helper import to_mask


def extract_edge(
    mask: np.ndarray,
    ksize: tuple[int, int] = (3, 3),
) -> np.ndarray:
    mask = to_mask(mask)

    if np.count_nonzero(mask) == 0:
        return mask

    kernel = np.ones(ksize, dtype=np.uint8)
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    edge = padded - cv2.erode(padded, kernel)
    edge = edge[1:-1, 1:-1]
    return (edge > 0).astype(np.uint8)
```

- [ ] **Step 4: Rewrite `mask/distance.py` to inline the pad helpers**

Replace the entire contents of `src/sdimg/mask/distance.py` with:

```python
import cv2
import numpy as np
from typing import Literal

from .helper import to_mask

DistanceType = Literal["l1", "l2", "c"]


def distance_transform(
    mask: np.ndarray,
    distance_type: DistanceType = "l2",
    mask_size: int = 3,
) -> np.ndarray:
    mask = to_mask(mask)

    if not np.any(mask):
        return np.zeros(mask.shape, dtype=np.float32)

    if distance_type == "l1":
        cv2_distance_type = cv2.DIST_L1
    elif distance_type == "l2":
        cv2_distance_type = cv2.DIST_L2
    elif distance_type == "c":
        cv2_distance_type = cv2.DIST_C
    else:
        raise ValueError("distance_type must be one of: 'l1', 'l2', 'c'.")

    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    distance = cv2.distanceTransform(
        padded,
        cv2_distance_type,
        mask_size,
        dstType=cv2.CV_32F,
    )
    return distance[1:-1, 1:-1]
```

- [ ] **Step 5: Delete `src/sdimg/mask/pad.py`**

Run: `rm src/sdimg/mask/pad.py`

- [ ] **Step 6: Verify no other file imports from `mask.pad`**

Run: `grep -rn "from .pad\|from sdimg.mask.pad\|mask\.pad" src/sdimg tests`
Expected: no results (empty output).

If any results appear, stop and update those files to inline as well.

- [ ] **Step 7: Run the test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add -u src/sdimg/mask/
git commit -m "refactor(mask): inline trivial 1px pad helpers and delete mask/pad.py"
```

---

### Task 3: `grabcut` uses `mask.distance_transform` instead of reimplementing it (F-03)

**Files:**
- Modify: `src/sdimg/fusion/grabcut.py`

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Replace the two inline `cv2.distanceTransform` calls in `_build_mask`**

In `src/sdimg/fusion/grabcut.py`, add this import alongside the other `..mask` import:

```python
from ..mask.distance import distance_transform
```

Then replace the `_build_mask` function body (lines beginning `def _build_mask(roi: np.ndarray) -> np.ndarray:`) with:

```python
def _build_mask(roi: np.ndarray) -> np.ndarray:
    din = distance_transform(roi)
    dout = distance_transform((1 - roi).astype(np.uint8))

    max_in = float(din.max())
    max_out = float(dout.max())

    th = min(max_in, max_out) / 5.0

    if th <= 0:
        mask = np.full(roi.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        mask[roi == 1] = cv2.GC_PR_FGD
        return mask

    mask = np.full(roi.shape, cv2.GC_BGD, dtype=np.uint8)
    mask[(roi == 0) & (dout < th)] = cv2.GC_PR_BGD
    mask[(roi == 1) & (din < th)] = cv2.GC_PR_FGD
    mask[din >= th] = cv2.GC_FGD

    return mask
```

Note: `(1 - roi).astype(np.uint8)` is required because `distance_transform` calls `to_mask`, which rejects non-binary inputs; `1 - roi` where `roi` is already `{0,1}` uint8 produces `{0,1}` uint8, but the explicit cast keeps dtype pinned.

- [ ] **Step 3: Verify no other inline `cv2.distanceTransform` calls remain in `fusion/`**

Run: `grep -rn "cv2\.distanceTransform" src/sdimg/fusion/`
Expected: no results (empty output).

- [ ] **Step 4: Run the test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass. In particular, the fusion contract tests should still pass — the padded `distance_transform` is semantically stricter (returns 0 at the real image border) but `grabcut` applies `cv2.copyMakeBorder(margin=20, value=0)` first, so the roi never touches the real border in practice. Results should be identical in typical use.

- [ ] **Step 5: Commit**

```bash
git add src/sdimg/fusion/grabcut.py
git commit -m "refactor(fusion): use mask.distance_transform in grabcut to honor layer order"
```

---

### Task 4: Replace `np.unique` value validation in `to_mask` with a single-pass check (F-09)

**Files:**
- Modify: `src/sdimg/mask/helper.py` (only the `to_mask` function)
- Test: `tests/mask/test_mask_contracts.py` (add edge-case tests first)

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Add edge-case tests for `to_mask` that pin the current behavior**

Append these tests to `tests/mask/test_mask_contracts.py`:

```python
def test_to_mask_accepts_all_zero_uint8() -> None:
    src = np.zeros((5, 5), dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.shape == (5, 5)
    assert int(out.sum()) == 0


def test_to_mask_accepts_all_one_uint8() -> None:
    src = np.ones((5, 5), dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 25


def test_to_mask_accepts_zero_and_255_uint8() -> None:
    src = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [1, 0]]


def test_to_mask_accepts_all_255_uint8() -> None:
    src = np.full((3, 3), 255, dtype=np.uint8)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert int(out.sum()) == 9


def test_to_mask_rejects_mixed_0_1_255() -> None:
    src = np.array([[0, 1, 255]], dtype=np.uint8)
    with pytest.raises(ValueError, match="binary values"):
        to_mask(src)


def test_to_mask_rejects_float_dtype_with_binary_values() -> None:
    # Current behavior: floats with {0.0, 1.0} are accepted because set subset
    # check succeeds. After refactor the behavior must be preserved.
    src = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    out = to_mask(src)
    assert out.dtype == np.uint8
    assert out.tolist() == [[0, 1], [1, 0]]
```

- [ ] **Step 3: Run the new tests against the current implementation to confirm baseline**

Run: `PYTHONPATH=src pytest tests/mask/test_mask_contracts.py -v -k "to_mask"`
Expected: all new tests pass on the current implementation.

- [ ] **Step 4: Rewrite `to_mask` in `src/sdimg/mask/helper.py`**

Replace the `to_mask` function (and only that function) in `src/sdimg/mask/helper.py` with:

```python
def to_mask(mask: object) -> np.ndarray:
    mask = ensure_mask(mask, name="Mask input")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    if mask.size == 0:
        return mask.astype(np.uint8)

    mx = mask.max()
    mn = mask.min()

    if mn >= 0 and mx <= 1:
        return mask.astype(np.uint8)

    if mx == 255 and mn >= 0:
        if bool(((mask == 0) | (mask == 255)).all()):
            return (mask > 0).astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, or {0, 255}.",
    )
```

Rationale (do not add as a comment — keep here for the reviewer):
- `bool` fast path unchanged.
- `size == 0` preserves acceptance of empty masks (current implementation accepts them via `set() <= {0,1}`).
- `mn >= 0 and mx <= 1` covers `{0}`, `{1}`, `{0,1}`, and also floats like `{0.0, 1.0}` — preserving current observed behavior pinned by `test_to_mask_rejects_float_dtype_with_binary_values`.
- `mx == 255` branch runs only if the first branch failed, and the `((m==0)|(m==255)).all()` confirms no stray values (e.g., rejects `{0, 1, 255}`).
- Single-pass scans replace `np.unique`'s sort.

- [ ] **Step 5: Run the full test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass, including the new edge-case tests.

- [ ] **Step 6: Commit**

```bash
git add src/sdimg/mask/helper.py tests/mask/test_mask_contracts.py
git commit -m "perf(mask): replace np.unique validation in to_mask with single-pass check"
```

---

### Task 5: Split `mask/helper.py` into `mask/convert.py` + `mask/bbox.py` (F-02, part 1)

**Files:**
- Create: `src/sdimg/mask/convert.py`
- Create: `src/sdimg/mask/bbox.py`
- Delete: `src/sdimg/mask/helper.py`
- Modify: `src/sdimg/mask/__init__.py`
- Modify: `src/sdimg/mask/morphology.py` (import path)
- Modify: `src/sdimg/mask/edge.py` (import path)
- Modify: `src/sdimg/mask/distance.py` (import path)
- Modify: `src/sdimg/mask/hull.py` (import path)
- Modify: `src/sdimg/mask/component.py` (import path)
- Modify: `src/sdimg/mask/hole.py` (import path)
- Modify: `src/sdimg/fusion/grabcut.py` (import path)

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Create `src/sdimg/mask/convert.py`**

Create the file with:

```python
import numpy as np

from .._core.validate import ensure_mask


def is_mask(mask: object) -> bool:
    try:
        to_mask(mask)
    except (TypeError, ValueError):
        return False

    return True


def to_mask(mask: object) -> np.ndarray:
    mask = ensure_mask(mask, name="Mask input")

    if mask.dtype == np.bool_:
        return mask.astype(np.uint8)

    if mask.size == 0:
        return mask.astype(np.uint8)

    mx = mask.max()
    mn = mask.min()

    if mn >= 0 and mx <= 1:
        return mask.astype(np.uint8)

    if mx == 255 and mn >= 0:
        if bool(((mask == 0) | (mask == 255)).all()):
            return (mask > 0).astype(np.uint8)

    raise ValueError(
        "Mask input must contain only binary values represented as "
        "bool, {0, 1}, or {0, 255}.",
    )
```

- [ ] **Step 3: Create `src/sdimg/mask/bbox.py`**

Create the file with:

```python
import numpy as np

from .._core.validate import ensure_ndarray
from .convert import to_mask


def get_coords(
    mask: np.ndarray,
    transpose: bool = False,
) -> np.ndarray:
    mask = to_mask(mask)
    coords = np.argwhere(mask > 0)

    if transpose:
        return coords.T

    return coords


def get_box_from_coords(
    coords: np.ndarray,
) -> tuple[int, int, int, int] | None:
    coords = ensure_ndarray(coords, name="coords")

    if coords.size == 0:
        return None

    if coords.ndim != 2:
        raise ValueError("coords must have shape (N, 2) or (2, N).")

    if coords.shape[1] == 2:
        h_coords = coords[:, 0]
        w_coords = coords[:, 1]
    elif coords.shape[0] == 2:
        h_coords = coords[0, :]
        w_coords = coords[1, :]
    else:
        raise ValueError("coords must have shape (N, 2) or (2, N).")

    hmin = int(np.min(h_coords))
    hmax = int(np.max(h_coords)) + 1
    wmin = int(np.min(w_coords))
    wmax = int(np.max(w_coords)) + 1
    return (wmin, hmin, wmax, hmax)


def get_box_from_mask(
    mask: np.ndarray,
) -> tuple[int, int, int, int] | None:
    mask = to_mask(mask)
    return get_box_from_coords(get_coords(mask))


def to_roi_box(
    mask: np.ndarray,
) -> dict[str, object] | None:
    mask = to_mask(mask)
    bbox = get_box_from_mask(mask)
    if bbox is None:
        return None

    wmin, hmin, wmax, hmax = bbox
    roi = mask[hmin:hmax, wmin:wmax].copy()
    return {"roi": roi, "box": bbox}


def get_roi_size(mask: np.ndarray) -> int:
    mask = to_mask(mask)
    return int(np.count_nonzero(mask))


def get_box_size(bbox: tuple[int, int, int, int] | None) -> int:
    if bbox is None:
        return 0

    wmin, hmin, wmax, hmax = bbox
    if wmin >= wmax or hmin >= hmax:
        raise ValueError("bbox must satisfy wmin < wmax and hmin < hmax.")
    return (wmax - wmin) * (hmax - hmin)


def get_centroid(
    mask: np.ndarray,
) -> tuple[float, float] | None:
    mask = to_mask(mask)
    coords = get_coords(mask)

    if coords.shape[0] == 0:
        return None

    center_h = float(np.mean(coords[:, 0]))
    center_w = float(np.mean(coords[:, 1]))
    return (center_h, center_w)
```

- [ ] **Step 4: Update `src/sdimg/mask/__init__.py` to re-export from the new files**

Replace the entire contents with:

```python
from .bbox import (
    get_box_from_coords,
    get_box_from_mask,
    get_box_size,
    get_centroid,
    get_coords,
    get_roi_size,
    to_roi_box,
)
from .component import pick_largest
from .convert import is_mask, to_mask
from .distance import distance_transform
from .edge import extract_edge
from .hole import fill_holes
from .hull import concave_hull, convex_hull
from .morphology import morphology
```

- [ ] **Step 5: Update internal imports from `.helper` to `.convert` / `.bbox`**

For each of these files, change the import line `from .helper import ...` as specified:

- `src/sdimg/mask/morphology.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/mask/edge.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/mask/distance.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/mask/hull.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/mask/component.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/mask/hole.py`: `from .helper import to_mask` → `from .convert import to_mask`
- `src/sdimg/fusion/grabcut.py`: replace the single line `from ..mask.helper import get_roi_size, to_mask` with two lines:
  ```python
  from ..mask.bbox import get_roi_size
  from ..mask.convert import to_mask
  ```

- [ ] **Step 6: Delete `src/sdimg/mask/helper.py`**

Run: `rm src/sdimg/mask/helper.py`

- [ ] **Step 7: Verify no dangling imports**

Run: `grep -rn "from .helper\|mask\.helper\|mask/helper" src/sdimg tests`
Expected: no results (empty output).

- [ ] **Step 8: Run the full test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass — no behavior change, only file layout.

- [ ] **Step 9: Commit**

```bash
git add -u src/sdimg/mask/ src/sdimg/fusion/grabcut.py
git add src/sdimg/mask/convert.py src/sdimg/mask/bbox.py
git commit -m "refactor(mask): split helper.py into convert.py and bbox.py"
```

---

### Task 6: Rename `image/helper.py` → `image/convert.py` (F-02, part 2)

**Files:**
- Create: `src/sdimg/image/convert.py`
- Delete: `src/sdimg/image/helper.py`
- Modify: `src/sdimg/image/__init__.py`
- Modify: `src/sdimg/image/norm.py` (import path)
- Modify: `src/sdimg/image/bc.py` (import path)
- Modify: `src/sdimg/image/denoise.py` (import path)
- Modify: `src/sdimg/spatial/patch.py` (import path)
- Modify: `src/sdimg/fusion/grabcut.py` (import path)
- Modify: `src/sdimg/fusion/otsu.py` (import path)

- [ ] **Step 1: Run the test suite to establish baseline green**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Create `src/sdimg/image/convert.py`**

Create the file with the exact contents that `image/helper.py` has today (verbatim copy):

```python
import numpy as np

from .._core.validate import ensure_image, ensure_ndarray


def is_image(image: object) -> bool:
    try:
        ensure_image(image, name="image")
    except (TypeError, ValueError):
        return False
    return True


def to_rgb(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")

    if image.ndim == 2:
        rgb = np.repeat(image[..., None], 3, axis=2)
        return to_uint8(rgb)

    channels = image.shape[2]
    if channels <= 2:
        rgb = np.repeat(image[..., 0:1], 3, axis=2)
    elif channels == 3:
        rgb = image
    else:
        rgb = image[..., :3]
    return to_uint8(rgb)


def to_gray(image: np.ndarray) -> np.ndarray:
    image = ensure_image(image, name="image")

    if image.ndim == 2:
        return to_uint8(image)

    channels = image.shape[2]
    if channels <= 2:
        gray = image[..., 0]
    else:
        gray = image[..., 0] * np.float32(0.299)
        gray += image[..., 1] * np.float32(0.587)
        gray += image[..., 2] * np.float32(0.114)

    return to_uint8(gray)


def to_uint8(image: np.ndarray) -> np.ndarray:
    image = ensure_ndarray(image, name="image")

    if image.dtype == np.uint8:
        return image

    if np.issubdtype(image.dtype, np.floating):
        clipped = np.clip(image, 0.0, 255.0)
        return np.rint(clipped).astype(np.uint8)

    return np.clip(image, 0, 255).astype(np.uint8)
```

- [ ] **Step 3: Update `src/sdimg/image/__init__.py`**

Replace the entire contents with:

```python
from .bc import adjust_brightness_contrast
from .blur import gaussian_blur, median_blur
from .convert import is_image, to_gray, to_rgb, to_uint8
from .denoise import denoise
from .norm import clahe_norm, hist_norm, minmax_norm, zscore_norm
from .sharpen import sharpen
```

- [ ] **Step 4: Update internal imports**

For each of these files, change the import as specified:

- `src/sdimg/image/norm.py`: `from .helper import to_uint8` → `from .convert import to_uint8`
- `src/sdimg/image/bc.py`: `from .helper import to_uint8` → `from .convert import to_uint8`
- `src/sdimg/image/denoise.py`: `from .helper import to_gray` → `from .convert import to_gray`
- `src/sdimg/spatial/patch.py`: `from ..image.helper import to_uint8` → `from ..image.convert import to_uint8`
- `src/sdimg/fusion/grabcut.py`: `from ..image.helper import to_gray` → `from ..image.convert import to_gray`
- `src/sdimg/fusion/otsu.py`: `from ..image.helper import to_gray` → `from ..image.convert import to_gray`

- [ ] **Step 5: Delete `src/sdimg/image/helper.py`**

Run: `rm src/sdimg/image/helper.py`

- [ ] **Step 6: Verify no dangling imports**

Run: `grep -rn "from .helper\|image\.helper\|image/helper" src/sdimg tests`
Expected: no results (empty output).

- [ ] **Step 7: Run the full test suite**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add -u src/sdimg/image/ src/sdimg/spatial/patch.py src/sdimg/fusion/
git add src/sdimg/image/convert.py
git commit -m "refactor(image): rename helper.py to convert.py"
```

---

### Task 7: Final verification sweep

- [ ] **Step 1: Run the full test suite once more**

Run: `PYTHONPATH=src pytest -q`
Expected: all tests pass.

- [ ] **Step 2: Verify no `helper.py` remains and no `mask/pad.py` remains**

Run: `find src/sdimg -name helper.py -o -name pad.py`
Expected output: only `src/sdimg/spatial/pad.py`.

- [ ] **Step 3: Verify the layer order is honored in `fusion/`**

Run: `grep -rn "cv2\.distanceTransform" src/sdimg/`
Expected output: only `src/sdimg/mask/distance.py` (the one canonical call site).

- [ ] **Step 4: Verify the public API is unchanged by importing everything**

Run: `PYTHONPATH=src .venv/bin/python -c "from sdimg.image import adjust_brightness_contrast, gaussian_blur, median_blur, denoise, is_image, to_gray, to_rgb, to_uint8, clahe_norm, hist_norm, minmax_norm, zscore_norm, sharpen; from sdimg.mask import distance_transform, concave_hull, convex_hull, extract_edge, get_box_size, get_roi_size, get_box_from_coords, get_box_from_mask, get_centroid, get_coords, is_mask, to_mask, to_roi_box, fill_holes, pick_largest, morphology; from sdimg.spatial import crop, pad_to_square, merge, split, resize, resize_keep_ratio, flip, rotate; from sdimg.fusion import grabcut, otsu_threshold; print('all public imports ok')"`
Expected output: `all public imports ok`.

- [ ] **Step 5: If everything above is green, the refactor is complete — no further commit needed.**

---

## Success criteria (from the spec)

1. ✅ F-01 / F-08: `mask/pad.py` no longer exists; the `pad` name is unambiguous at the `spatial` layer. (Task 2, verified in Task 7 Step 2)
2. ✅ F-02: `mask/convert.py`, `mask/bbox.py`, `image/convert.py` each hold one concern. (Tasks 5, 6)
3. ✅ F-03: `grabcut._build_mask` consumes `mask.distance_transform`; only one `cv2.distanceTransform` call site remains. (Task 3, verified in Task 7 Step 3)
4. ✅ F-05 / F-06: README and `CLAUDE.md` state the RGB contract and channel-count semantics. (Task 1)
5. ✅ F-09: `to_mask` uses a single-pass check. (Task 4)
