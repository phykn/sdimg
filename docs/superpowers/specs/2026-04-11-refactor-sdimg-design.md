# Refactor Diagnosis — sdimg (2026-04-11)

## Scope

Full sweep of `src/sdimg/` (30 files, ~1200 LOC). Modules: `_core`, `image`, `mask`, `spatial`, `fusion`. Declared layer order: `_core ← image ← mask ← spatial ← fusion`.

## Context summary

Sources read:
- `CLAUDE.md`, `README.md`
- Full source tree under `src/sdimg/`
- Test file locations under `tests/` (contract-style tests per module)
- Recent `git log` (last 13 commits, current version 0.2.0)

Two falsification experiments were run:
1. **F-03** — compared `cv2.distanceTransform` with/without 1px zero padding on a border-touching mask (20×20, mask occupies top-left 10×10). Max abs difference: **8.60**. Confirmed the duplication in `grabcut._build_mask` silently drifts from `mask/distance.py` semantics.
2. **F-09** — benchmarked `set(np.unique(m).tolist())` vs `((m==0)|(m==1)).all()` on a 4096×4096 uint8 mask. Result: **197.4 ms vs 27.7 ms (7.12×)**. Confirmed real per-call cost.

## Selected findings

### F-01: `pad.py` name collision between layers

- **Location**: `src/sdimg/mask/pad.py`, `src/sdimg/spatial/pad.py`
- **Category**: structure
- **Observation**: Two files share the name `pad.py`. `mask/pad.py` is private (`_pad_1px`/`_unpad_1px`) used only by `morphology.py`, `edge.py`, `distance.py`. `spatial/pad.py` is the public API (`pad_to_square`).
- **Reconstruction attempt**: "Padding utilities should live in one place."
- **Failure point**: The file name collision destroys the MECE partition — a reader searching for "padding" has no way to decide which file to open, and the private helper file has no real relation to the public `pad` concept.
- **Suggested direction**: Remove `mask/pad.py` entirely (see F-08) so `spatial/pad.py` becomes the unambiguous home for the `pad` name.
- **Axes**: Impact: med, Confidence: high, Effort: S

### F-02: `helper.py` as a catch-all

- **Location**: `src/sdimg/image/helper.py` (59 LOC), `src/sdimg/mask/helper.py` (120 LOC)
- **Category**: naming
- **Observation**: `mask/helper.py` mixes three concerns: (a) validation/conversion (`is_mask`, `to_mask`), (b) coordinate extraction (`get_coords`, `get_box_from_coords`), (c) bbox/centroid geometry (`get_box_from_mask`, `to_roi_box`, `get_box_size`, `get_roi_size`, `get_centroid`). `image/helper.py` mixes validation (`is_image`), channel conversion (`to_gray`, `to_rgb`), and dtype conversion (`to_uint8`).
- **Reconstruction attempt**: "A grab-bag of utilities used across the module."
- **Failure point**: `helper` is not a concern — it is the absence of one. The partition is unnamed, so the module boundary cannot be reconstructed from the file name.
- **Suggested direction**:
  - `mask/helper.py` → split into `mask/convert.py` (`is_mask`, `to_mask`) and `mask/bbox.py` (coords + box + centroid + roi helpers).
  - `image/helper.py` → `image/convert.py` (all four functions fit one concern: "convert an image into a target representation").
- **Axes**: Impact: med, Confidence: high, Effort: M

### F-03: `grabcut` reimplements distance transform, bypassing `mask.distance_transform`

- **Location**: `src/sdimg/fusion/grabcut.py:53-54` (`_build_mask`)
- **Category**: duplication
- **Observation**: `_build_mask` calls `cv2.distanceTransform(roi, cv2.DIST_L2, 3, dstType=cv2.CV_32F)` directly with no padding. `mask/distance.py::distance_transform` pads 1px before computing distance and unpads after. `fusion` importing `cv2` to re-do a primitive that `mask` already exposes violates the stated layer order (`fusion ← mask`).
- **Reconstruction attempt**: "Compute inside/outside distance fields to drive the trimap threshold."
- **Failure point**: Two problems.
  1. **Architectural**: `fusion/grabcut.py` must consume `mask/distance.py`, not reimplement it.
  2. **Latent semantic drift**: experiment showed an 8.6-unit max abs difference at the border row when the mask touches the image edge. `grabcut` currently masks this bug because `cv2.copyMakeBorder(..., value=0)` with `margin ≥ 1` is applied before `_build_mask`, so `roi` never touches the crop border in practice. But the invariant is not stated, and any future change loosening the margin requirement (or a caller skipping the margin) would surface the bug.
- **Suggested direction**: Import `distance_transform` from `sdimg.mask.distance`. Replace the two inline `cv2.distanceTransform(...)` calls in `_build_mask` with `distance_transform(roi)` and `distance_transform(1 - roi)`.
- **Axes**: Impact: high, Confidence: high, Effort: S
- **Verification**: `PYTHONPATH=src .venv/bin/python -c "..."` on a 20×20 border-touching mask → max abs diff 8.60, border row unpadded `[9.55, 8.60, 7.64, ...]` vs padded `[0.96, 0.96, 0.96, ...]`. Confirmed.

### F-05: `to_gray` assumes RGB channel order (no enforcement, no documentation)

- **Location**: `src/sdimg/image/helper.py:41-43`
- **Category**: logic
- **Observation**: `to_gray` applies Rec. 601 weights `0.299*c[0] + 0.587*c[1] + 0.114*c[2]`. `ensure_image` validates shape only. Neither `_core` nor README states which channel order sdimg expects. Other modules implicitly assume RGB as well (`clahe_norm` and `hist_norm` use `COLOR_RGB2YCrCb`).
- **Reconstruction attempt**: "Input images are in RGB."
- **Failure point**: The invariant is load-bearing for correctness but has no home. `cv2.imread` returns BGR by default; a user feeding such an array gets silently wrong luminance (R and B weights swapped).
- **User decision (2026-04-11)**: **Force RGB.** sdimg contracts commit to RGB-ordered input for color images.
- **Suggested direction**:
  - Add an explicit "Input color images must be in RGB channel order" line to `CLAUDE.md`, `README.md` (Core Contracts section), and `_core/validate.py::ensure_image` docstring.
  - Do not add a runtime check (byte-level detection is unreliable) — this is a documented contract, enforced at the boundary by the caller.
- **Axes**: Impact: med, Confidence: high (after decision), Effort: S

### F-06: `to_gray`/`to_rgb` silently assume C=2 is Gray+Alpha and C=4 is RGBA

- **Location**: `src/sdimg/image/helper.py:22, 37-38`
- **Category**: logic
- **Observation**: For C=2, both `to_gray` and `to_rgb` take `image[..., 0]` and drop channel 1. For C=4, `to_rgb` takes `image[..., :3]` and drops channel 3.
- **Reconstruction attempt**: "C=2 means Gray+Alpha, C=4 means RGBA; alpha is discarded."
- **Failure point**: The assumption is reasonable but undocumented. A 2-channel gradient image would be silently misinterpreted.
- **Suggested direction** (bundled with the RGB policy in F-05): document the channel-count semantics in the same contract section:
  - C=1: grayscale
  - C=2: grayscale + alpha (alpha ignored)
  - C=3: RGB
  - C=4: RGBA (alpha ignored)
- **Axes**: Impact: low, Confidence: high (after decision), Effort: S

### F-08: `_pad_1px` / `_unpad_1px` are unnecessary abstractions

- **Location**: `src/sdimg/mask/pad.py`
- **Category**: complexity
- **Observation**: 5-line and 1-line wrappers around `np.pad(mask, ((1,1),(1,1)), constant_values=0)` and `mask[1:-1, 1:-1]`. Used by 3 call sites (`morphology.py`, `edge.py`, `distance.py`).
- **Reconstruction attempt**: "Extracted to DRY up the 1px-pad pattern."
- **Failure point**: The wrapper is not shorter than the inline form — `np.pad(mask, 1, constant_values=0)` (numpy accepts an int for uniform padding) and `mask[1:-1, 1:-1]` are both short, explicit, and localized. The abstraction adds a file, an import, and a layer of indirection to save nothing.
- **Suggested direction**: Inline both helpers at all three call sites. Delete `src/sdimg/mask/pad.py`. Simultaneously resolves F-01.
- **Axes**: Impact: low, Confidence: high, Effort: S

### F-09: `to_mask` uses `np.unique` for value validation (7× slower than single-pass)

- **Location**: `src/sdimg/mask/helper.py:21-27`
- **Category**: perf-waste
- **Observation**: Current implementation:
  ```python
  unique_values = set(np.unique(mask).tolist())
  if unique_values <= {0, 1}: ...
  if unique_values <= {0, 255}: ...
  ```
  Every mask function calls `to_mask` at entry, so this cost compounds across a pipeline.
- **Reconstruction attempt**: "Accept bool, `{0,1}`, `{0,255}` encodings and normalize to `{0,1}` uint8."
- **Failure point**: `np.unique` sorts the whole array (`O(n log n)`). A single-pass check `((m==0)|(m==1)).all()` is `O(n)` and short-circuits in C. Benchmark on 4096×4096 uint8 confirmed **197.4 ms → 27.7 ms (7.12×)**.
- **Suggested direction**: Rewrite as a dtype-first dispatch:
  1. `bool` → `.astype(np.uint8)`.
  2. Otherwise compute `mx = int(mask.max())` (single pass).
  3. If `mx <= 1` and `mask.min() >= 0` → return as uint8 (already `{0,1}` or `{0}`).
  4. If `mx == 255` and `((mask==0) | (mask==255)).all()` → return `(mask > 0).astype(np.uint8)`.
  5. Else raise `ValueError`.
  Must preserve the current error message for the `else` branch. Must preserve acceptance of empty/all-zero/all-one masks.
- **Axes**: Impact: med, Confidence: high, Effort: S
- **Verification**: 4096×4096 uint8 benchmark — np.unique 197.4 ms vs single-pass 27.7 ms (7.12×). Confirmed.

## Refactoring constraints

- **All existing tests must pass** (`PYTHONPATH=src pytest -q`).
- **Public API surface must remain stable.** No renames, removals, or signature changes for anything re-exported from `sdimg.image`, `sdimg.mask`, `sdimg.spatial`, `sdimg.fusion`. F-02's module splits must keep the `mask/__init__.py` and `image/__init__.py` re-exports intact so `from sdimg.mask import to_mask` still works.
- **Error policy preserved.** F-09's rewrite of `to_mask` must still raise `ValueError` on non-binary inputs with a message covering `bool, {0, 1}, {0, 255}`.
- **Layer order preserved.** F-03 must import from `sdimg.mask.distance` — do not create a sideways dependency.
- **No performance regressions** in mask pipelines. F-09 should improve; F-03 should be neutral-to-better (one more function call, same underlying cv2 op).
- **No new dependencies.**

## Success criteria

After the refactor, re-running the Feynman reconstruction on the changed areas should succeed where it previously failed:

1. **F-01 / F-08**: `mask/pad.py` no longer exists; the `pad` name is unambiguous at the `spatial` layer.
2. **F-02**: Asked "what is in `mask/convert.py`?" the answer is "mask-input validation and normalization" — one concern per file. Same for `mask/bbox.py` and `image/convert.py`.
3. **F-03**: Asked "where is distance transform computed?" the answer is `mask/distance.py` — one place, one implementation. `grabcut._build_mask` is a consumer, not a reimplementer.
4. **F-05 / F-06**: README and `CLAUDE.md` state the RGB contract and channel-count semantics explicitly. Asked "what channel order does sdimg expect?" a new reader can answer from docs alone.
5. **F-09**: Asked "why is `to_mask` using a single-pass check instead of `np.unique`?" the answer is "to avoid an `O(n log n)` sort on every mask operation — measured 7× faster." The why is preserved in commit history / PR description, not as a comment.

## Out of scope (deferred)

- **F-04** (binary uint8 gap): low impact, revisit if a new mask function needs to be added.
- **F-07** (patch axis brute-force): medium effort, working correctly — revisit if patch counts grow large.
- **F-10** (morphology ops half-dispatch): cosmetic.
- **F-11** (grabcut `/5.0` magic constant): needs a domain decision, not a refactor.
- **F-12** (merge function mixing concerns): low impact, large function.
