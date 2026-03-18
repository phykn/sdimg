# sdimg 프로젝트 사양서 (Specification)

이 문서는 `sdimg` 라이브러리의 핵심 아이디어, 데이터 계약, 기능 범위를 정리한 사양 문서입니다. 구현 단계에서 해석이 갈리지 않도록 공통 규칙과 모듈별 책임을 우선 정의합니다.

---

## 1. 프로젝트 개요
*   **프로젝트 명**: `sdimg`
*   **프로젝트 성격**: 기존의 맥락이나 라이브러리를 따르지 않는 **완전한 신규(Greenfield) 개발**.

## 2. 공통 데이터 계약 (Core Data Contract)

구현 전반에서 아래 규칙을 공통 계약으로 간주합니다.

### 2.1 입력 범위
*   **입력 타입**: 모든 함수 입력은 `np.ndarray`만 허용.
*   **이미지 입력**: grayscale 또는 color 입력을 허용.
*   **마스크 입력**: 마스크는 `0/1`, `0/255`, `bool`, `0.0/1.0`처럼 명확한 바이너리 표현만 허용. 확률 맵, 가중치 맵, 연속값 마스크는 범위 밖으로 본다.

### 2.2 내부 표준화 규칙
*   **이미지 표준화**: 모든 입력은 `to_rgb` 또는 `to_gray`를 거쳐 채널 표현을 정규화하고 `np.uint8`로 강제 변환.
    - `to_gray`의 결과 shape는 `(H, W)`로 고정.
*   **마스크 표준화**: 모든 마스크는 `to_mask`를 거쳐 내부 연산 전에 `{0, 1}` 값으로 강제 정규화.
    - `to_mask`는 `0/1`, `0/255`, `bool`, `0.0/1.0`처럼 명확한 바이너리 표현만 허용.
*   **정밀도 규칙**: 필터링, 정규화, 보간 등 중간 계산은 `float32` 이상 사용을 허용.

### 2.3 출력 규칙
*   **이미지 반환 규칙**: 이미지 전처리 결과는 항상 `np.uint8`, `0~255` 범위로 반환.
*   **마스크 반환 규칙**: 마스크 처리 결과는 항상 `{0, 1}` 값의 `np.uint8` 바이너리 마스크로 반환.
*   **예외 함수**: `distance transform`은 예외적으로 연속값 거리 맵을 반환할 수 있음.
*   **좌표계 규칙**: 바운딩 박스는 항상 `(wmin, hmin, wmax, hmax)` 순서를 사용.
    - `wmax`, `hmax`는 포함하지 않는 half-open 규칙으로 정의.
*   **좌표 목록 규칙**: 픽셀 좌표는 `(h, w)` 순서를 사용.

### 2.4 오류 처리 원칙
*   **형식 정규화 우선**: grayscale/color 차이와 채널 표현 차이는 `to_rgb`, `to_gray`를 통해 우선 정규화하고, 이미지 dtype은 `np.uint8`로 맞춤.
*   **형식 오류**: 정규화로 수용할 수 없는 비정상 shape, 채널 구성, 또는 binary로 해석 불가능한 값 표현은 예외 처리.
*   **입력 타입 오류**: `np.ndarray` 이외 입력은 자동 로드/변환하지 않고 예외 처리.
*   **빈 마스크 처리**: 객체가 없는 마스크를 받았을 때의 동작은 함수별로 명시.
*   **shape 불일치**: image-mask pair 연산에서 shape이 다르면 자동 보정하지 않고 예외 처리.

## 3. 기능적 요구사항 (Functional Requirements)

기능의 일관성과 확장성을 위해 다음 네 가지 핵심 카테고리로 모듈화하여 제안합니다.

### 3.0 공통 표준 (Foundation)
*   **표준화 함수 제공**: 이미지/마스크 입력 검증 및 정규화를 공통 유틸리티로 분리.
*   **출력 정리**: 내부 계산 결과를 사용자 API의 공통 반환 규칙에 맞게 정리.
*   **메타데이터 원칙**: 메타데이터는 필요한 함수에서만 선택적으로 반환.

### 3.1 이미지 전처리 (Image-Specific)
픽셀 값 자체를 수정하여 특징을 강화하거나 노이즈를 제어하는 단계입니다.

*   **채널 처리 규칙**: 별도 명시가 없으면 모든 이미지 전처리 연산은 RGB 각 채널에 독립적으로 적용.
*   **입출력 규칙**: 입력은 표준화 단계에서 `np.uint8`로 맞추고, 출력도 항상 `np.uint8`, `0~255`로 반환.

#### 3.1.1 정규화 및 대비 정밀 개선 (Normalization & Contrast)
- **standard normalization (`standard_norm`)**: 평균과 표준편차를 활용해 정규화한 뒤, 지정한 표준편차 범위 기준으로 값을 잘라 `0~255` 범위로 재매핑하여 반환.
- **histogram normalization (`hist_norm`)**: 히스토그램 상한/하한을 활용한 전역 대비 개선.
- **CLAHE normalization (`clahe_norm`)**: 국소적 대비를 개선하여 세부 특징 부각.

#### 3.1.2 기본 픽셀 수준 변환 (Point-wise Transformation)
- **brightness adjustment**: 전역 밝기 보정.
    - `brightness` 범위는 `[-1, 1]`.
    - `0`은 원본 유지, `-1`은 전체 검정, `1`은 전체 흰색을 의미.
- **contrast adjustment**: 전역 대비 보정.
    - `contrast` 범위는 `[-1, 1]`.
    - `0`은 원본 유지, `-1`은 전체 중간 회색(`128`)에 수렴, `1`은 대비를 최대로 강화하는 방향을 의미.
    - 대비 연산의 중심값은 `128`을 기준으로 함.
    - `contrast < 0`이면 픽셀 값을 `128` 쪽으로 수축하고, `contrast > 0`이면 `128`에서 멀어지도록 확장.
    - 정확한 배율 함수는 구현 단계에서 별도로 확정.
    - 밝기와 대비를 함께 적용할 때의 순서는 `brightness -> contrast`로 고정.

#### 3.1.3 필터링 및 노이즈 제거 (Filtering & Smoothing)
- **blurring**: 이미지의 노이즈를 제거하거나 디테일을 부드럽게 만드는 연산 (Gaussian, Median, Bilateral 등 지원).
- **denoising (NL-means)**: Non-Local Means Denoising 기법을 활용한 고품질 노이즈 제거. 지엽적인 특성뿐만 아니라 이미지의 전역적인 유사성을 고려하여 구조를 보존하면서 노이즈를 획기적으로 줄림.
- **정밀도/출력 dtype 규칙**: 내부 계산은 `float32` 이상을 허용하되, 최종 출력은 항상 `np.uint8`, `0~255`로 정리하여 반환.

#### 3.1.4 세부 특징 강화 (Detail Enhancement)
- **sharpening**: 이미지의 경계선을 강조하여 선명도를 높이는 연산 (Unsharp Mask, Laplacian 필터 등 지원).

### 3.2 마스크 처리 (Mask-Specific)
관심 영역(ROI)을 정의하는 **바이너리 마스크 `{0, 1}`** 의 품질을 개선하고 정보를 추출하는 단계입니다.

*   **입력 규칙**: 모든 마스크 입력은 `to_mask`를 거쳐 `{0, 1}` 바이너리 마스크로 정규화.
*   **빈 마스크 규칙**: `bounding box calculation`, `centroid extraction`은 빈 마스크에서 `None`을 반환.

#### 3.2.1 마스크 정제 및 형태 변환 (Refinement & Morphology)
- **morphology transformation**: 침식(Erosion), 팽창(Dilation), 열기(Opening), 닫기(Closing) 연산을 통해 마스크의 노이즈를 제거하거나 외곽선을 정제.
    - 기본 구조 요소는 사각형 커널로 정의.
    - 커널 크기는 조절 가능하도록 설계.
- **hole filling**: 마스크 내부의 고립된 빈 공간(Hole)을 찾아 자동으로 메워주는 기능.
- **keep largest component**: 마스크 내 여러 개의 독립된 영역(Connected Components)이 존재할 때, 면적이 가장 큰 영역 하나만 남기고 나머지는 제거하는 정제 기능.
    - 동일 면적 component가 여러 개면 index가 가장 작은 component를 유지.

#### 3.2.2 외곽선 및 기하학적 피팅 (Boundary & Fitting)
- **convex hull**: 마스크의 가장 바깥쪽 점들을 연결하여 볼록한 다각형(Convex Hull)을 생성하고 내부를 채움.
    - 반환은 `{0, 1}` 값의 `np.uint8` 바이너리 마스크.
    - 빈 마스크에서는 same shape의 all-zero 마스크를 반환.
- **concave hull**: 마스크의 형상을 더 세밀하게 따라가는 오목 형태(Alpha Shape 등)를 생성하여 객체 외곽선을 정립.
    - 형상 제어 파라미터(예: `alpha`)를 받는 방식으로 정의.
    - 반환은 `{0, 1}` 값의 `np.uint8` 바이너리 마스크.
    - 빈 마스크에서는 same shape의 all-zero 마스크를 반환.
- **edge extraction (Boundary)**: 마스크 내부 객체의 외곽선(Boundary/Contour)만을 추출하여 `{0, 1}` 바이너리 마스크로 반환.
    - 빈 마스크에서는 same shape의 all-zero 마스크를 반환.

#### 3.2.3 분석 및 정보 추출 (Analysis & Extraction)
- **distance transform**: 마스크 내부의 각 픽셀에서 가장 가까운 배경(0)까지의 거리를 계산하여 수치화.
    - 반환은 예외적으로 연속값 거리 맵이며, 기본 dtype은 `float32`.
    - 기본 거리 기준은 Euclidean (`L2`).
    - 빈 마스크에서는 same shape의 all-zero `float32` 맵을 반환.
- **coordinate extraction**: 마스크 내에서 '1'인 값을 가진 모든 픽셀의 좌표$(h, w)$ 목록을 추출.
    - 반환은 shape `(N, 2)`의 `np.ndarray`.
    - 빈 마스크에서는 shape `(0, 2)`의 빈 `np.ndarray`를 반환.
- **bounding box calculation**: 객체를 감싸는 최소 사각형의 위치$(wmin, hmin, wmax, hmax)$ 및 크기$(w, h)$를 산출.
    - 빈 마스크에서는 `None`을 반환.
- **area measurement**: 마스크 영역의 전체 픽셀 수를 계산.
    - 반환은 `int`.
    - 빈 마스크에서는 `0`을 반환.
- **centroid extraction**: 객체의 기하학적인 무게 중심$(ch, cw)$ 좌표를 추출.
    - 반환 좌표는 `float`.
    - 빈 마스크에서는 `None`을 반환.

### 3.3 공간 변환 (Spatial Transformation)
이미지 또는 마스크 개별 데이터의 외형이나 위치를 기하학적으로 변경하거나 데이터를 조각화하는 단계입니다. (각 데이터는 **독립적으로** 처리됨)

#### 3.3.1 기하학적 변형 및 해상도 조정 (Geometry & Resolution)
- **resizing**: 데이터의 해상도를 조정하는 연산.
    - 크기 인자는 `height=`, `width=`를 분리하여 받음.
    - `height`, `width`를 모두 지정하면 해당 크기로 변환.
    - 둘 중 하나만 지정하면 aspect ratio를 유지하면서 나머지 크기를 자동 계산.
    - 둘 다 `None`이면 예외 처리.
    - **Aspect Ratio Preservation**: Letterboxing, Scaling 등 비율 보존 옵션 지원.
    - **Interpolation**: 이미지와 마스크는 기본 보간 정책을 분리함. 이미지의 기본 보간은 `cubic`, 마스크의 기본 보간은 `nearest_exact`.
- **rotation/flip**: 데이터 증강 및 정위치 정렬을 위한 공간 연산.
    - `rotation`은 90도 단위의 양수 각도만 허용하며, 기본 허용값은 `0`, `90`, `180`, `270`.
    - `flip`은 `horizontal`, `vertical`, `transpose`를 지원.
    - `transpose`는 대각선 기준 전치로 정의.

#### 3.3.2 여백 및 영역 조정 (Padding & Area Adjustment)
- **padding / unpadding**: 데이터의 외곽에 공백을 채우거나 원래 크기로 복원하는 연산 (Constant, Reflect, Edge 등 지원).
    - `padding`은 단일 정수 인자를 받아 상하좌우에 동일 크기로 적용.
    - 기본 모드는 이미지에서 `mirror`, 마스크에서 `constant`.
    - 마스크의 `constant` padding 기본값은 `0`.
    - 필요 시 `return_meta=True`로 `unpadding`에 필요한 pad 정보를 함께 반환.
    - `unpadding`은 pad 정보를 받아 원래 영역을 복원.
- **cropping**: 바운딩 박스 좌표$(wmin, hmin, wmax, hmax)$를 입력으로 받아 특정 영역을 추출하고 데이터 크기를 축소.
    - bbox는 half-open 규칙을 따르며 `img[hmin:hmax, wmin:wmax]`와 동일한 의미로 해석.
    - 범위를 벗어나거나 순서가 잘못된 bbox는 예외 처리.

#### 3.3.3 데이터 조각화 및 복원 (Partitioning & Reconstruction)
- **split (Patchify)**: 단일 입력(이미지 혹은 마스크)을 $n \times n$ 격자로 조각화.
    - `n=1`이면 분할하지 않음.
    - 모든 patch는 동일 크기를 유지.
    - `overlap`은 비율 인자로 받고, 범위는 `0 <= overlap < 1`.
    - 예: `overlap=0.2`면 인접 patch가 최소 20% 정도 겹치도록 조정.
    - 가능한 한 최소 overlap 이상을 만족하되, 그중 overlap이 가장 작아지도록 patch 시작점을 자동 계산.
    - 필요 시 `return_meta=True`로 `merge`에 필요한 메타데이터를 함께 반환.
- **merge (Stitch)**: 쪼개진 패치 리스트를 다시 하나의 데이터로 복원.
    - `split`에서 생성된 patch와 대응 메타데이터를 받아 원본 크기로 복원.
    - 이미지의 기본 블렌딩은 코사인 블렌딩(Cosine Blending).
    - 이미지 복원 결과는 최종적으로 `np.uint8`, `0~255`로 정리하여 반환.
    - 마스크의 기본 복원 규칙은 overlap 영역에서 `logical OR`.

### 3.4 이미지-마스크 융합 처리 (Image-Mask Fusion)
이미지와 마스크 데이터를 동시에 입력으로 받아 연동하여 처리하거나, 이미지로부터 마스크를 생성/보정하는 단계입니다.

#### 3.4.1 이미지 기반 마스크 생성 (Image-based Mask Generation)
- **otsu thresholding**: 이미지의 히스토그램을 분석하여 최적의 임계값(Threshold)을 자동으로 계산하고 초기 마스크({0, 1})를 생성.
    - 반환은 `{0, 1}` 값의 `np.uint8` 바이너리 마스크.

#### 3.4.2 이미지-마스크 가이디드 보정 (Mask-guided Refinement)
- **grabcut refinement**: 이미지의 색상 정보와 초기 마스크 힌트를 결합하여 객체 경계를 반복적 그래프 컷 알고리즘으로 분리.
    - 입력은 `image + initial mask`를 필수로 받음.
    - 최종 반환은 `{0, 1}` 값의 `np.uint8` 바이너리 마스크.
- **guided filter refinement**: 원본 이미지의 엣지 정보를 가이드로 삼아 마스크 경계를 이미지의 실제 객체 외곽선에 밀착시키는 정밀 필터링 기법.
    - 입력은 `image + initial mask`를 필수로 받음.
    - 내부 계산은 연속값을 허용하되, 최종 반환은 `{0, 1}` 바이너리 마스크로 고정.
- **active contour refinement (Snakes)**: 마스크 경계를 물리적으로 변형하여 이미지의 가장 선명한 가장자리(Edge)를 향해 수축/팽창하며 정착시키는 기법.
    - 입력은 `image + initial mask`를 필수로 받음.
    - 최종 반환은 `{0, 1}` 값의 `np.uint8` 바이너리 마스크.

#### 3.4.3 동기화된 공간 변환 (Synchronized Transformations)
- **synchronized resizing**: 이미지와 마스크를 동일한 비율과 보간 방식으로 함께 리사이즈.
    - 기본 보간은 이미지 `cubic`, 마스크 `nearest_exact`를 따름.
- **synchronized rotation/flip**: 동일한 각도 및 방향으로 회전/뒤집기 수행.
    - `rotation`은 `0`, `90`, `180`, `270`만 허용.
    - `flip`은 `horizontal`, `vertical`, `transpose`를 지원.
- **synchronized padding / unpadding**: 이미지와 마스크 외곽에 동일한 크기의 여백을 추가하거나 원래 크기로 복원.
    - `padding`은 단일 정수 인자를 받아 상하좌우에 동일 크기로 적용.
    - 기본 모드는 이미지에서 `mirror`, 마스크에서 `constant`.
    - 마스크의 `constant` padding 기본값은 `0`.
    - 필요 시 `return_meta=True`로 `unpadding`에 필요한 pad 정보를 함께 반환.
    - `unpadding`은 pad 정보를 받아 원래 영역을 복원.
- **synchronized cropping**: 바운딩 박스 좌표$(wmin, hmin, wmax, hmax)$로부터 이미지와 마스크의 동일한 영역을 한 번에 추출.
    - bbox는 half-open 규칙을 따르며 `img[hmin:hmax, wmin:wmax]`와 동일한 의미로 해석.
    - 범위를 벗어나거나 순서가 잘못된 bbox는 예외 처리.
- **synchronized split (Patchify)**: 이미지와 마스크를 동일한 패치 크기 및 중첩(Overlap) 규칙으로 조각화하여 한 쌍의 패치 리스트를 구성.
    - `n=1`이면 분할하지 않음.
    - 모든 patch는 동일 크기를 유지.
    - `overlap`은 비율 인자로 받고, 범위는 `0 <= overlap < 1`.
    - 가능한 한 최소 overlap 이상을 만족하되, 그중 overlap이 가장 작아지도록 patch 시작점을 자동 계산.
    - 필요 시 `return_meta=True`로 `merge`에 필요한 메타데이터를 함께 반환.
- **synchronized merge (Stitch)**: 쪼개진 이미지와 마스크 패치들을 각각 동일한 블렌딩 옵션을 적용하여 다시 복원.
    - `split`에서 생성된 patch와 대응 메타데이터를 받아 원본 크기로 복원.
    - 이미지의 기본 블렌딩은 코사인 블렌딩(Cosine Blending).
    - 이미지 복원 결과는 최종적으로 `np.uint8`, `0~255`로 정리하여 반환.
    - 마스크의 기본 복원 규칙은 overlap 영역에서 `logical OR`.

## 4. API 및 구현 원칙

### 4.1 API 형태
*   **우선 형태**: 모듈 함수 중심 API를 기본으로 함. 예: `sdimg.image.hist_norm(...)`
*   **최상위 노출**: 자주 쓰는 함수는 `sdimg.hist_norm(...)` 형태로 재노출 가능.
*   **일관된 시그니처**: 동일 계열 함수는 `src`, 옵션 인자, 반환 형식을 최대한 통일.
*   **입력 이름 규칙**: 단일 입력 함수는 `src`, image-mask pair 함수는 `image, mask` 또는 `image, initial_mask`를 기본으로 사용.
*   **이름 규칙**: helper 성격의 조회 함수는 `get_*` 형식을 우선 사용하고, 알고리즘 함수는 `distance_transform`, `convex_hull`처럼 의미 중심 이름을 유지.

### 4.2 반환 정책
*   **기본 반환값**: 기본 반환은 처리 결과만 반환하도록 통일.
*   **메타데이터 반환**: 메타데이터가 필요한 함수는 `return_meta=True` 같은 별도 인자를 통해 요청했을 때만 추가 반환을 허용.
*   **pair 연산**: image-mask 동시 연산의 기본 반환은 `image, mask` 순서를 따른다.
*   **pair + meta 반환**: 메타데이터를 함께 반환할 때는 `image, mask, meta`처럼 평평한 순서를 따른다.

### 4.3 의존성 원칙
*   **핵심 의존성**: `numpy`를 기본 전제로 둠.
*   **선택 의존성**: `opencv`, `scipy`, `scikit-image` 등은 기능별 선택 의존성으로 분리 가능.
*   **대체 구현 고려**: 특정 외부 라이브러리에 강하게 종속되는 기능은 대체 경로 또는 graceful failure를 고려.

### 4.4 테스트 기준
*   **정확성**: shape, dtype, 값 범위, 바이너리 마스크 유지 여부를 검증.
*   **경계조건**: 빈 입력, 단일 픽셀 객체, 채널 수 불일치, 잘못된 bbox 입력을 검증.
*   **재현성**: 난수 또는 반복 최적화 기반 알고리즘은 seed 또는 deterministic 옵션을 고려.

## 5. 프로젝트 구조 (Project Structure)

사양서의 기능적 요구사항을 모듈별로 관리하기 위한 권장 프로젝트 레이아웃입니다.

```text
sdimg/
├── __init__.py      # 최상위 API 노출 (sdimg.hist_norm() 등)
├── core.py          # 공통 표준화, 검증, 공통 반환 형식 정리
├── image/
│   ├── __init__.py      # 이미지 관련 상위 API 재노출
│   ├── standard_norm.py # standard normalization
│   ├── hist_norm.py     # histogram normalization
│   ├── clahe.py         # CLAHE normalization
│   ├── brightness.py    # brightness adjustment
│   ├── contrast.py      # contrast adjustment
│   ├── blur.py          # blurring
│   ├── denoise.py       # NL-means denoising
│   └── sharpen.py       # sharpening
├── mask/
│   ├── __init__.py  # 마스크 관련 상위 API 재노출
│   ├── morphology.py    # morphology transformation
│   ├── hole.py          # hole filling
│   ├── component.py     # keep largest component
│   ├── hull.py          # convex hull, concave hull
│   ├── edge.py          # edge extraction
│   ├── distance.py      # distance transform
│   └── helper.py        # get_coords, get_bbox, get_area, get_centroid
├── spatial/
│   ├── __init__.py      # 공간 변환 관련 상위 API 재노출
│   ├── resize.py        # resizing
│   ├── transform.py     # rotation, flip, transpose
│   ├── pad.py           # padding, unpadding
│   ├── crop.py          # cropping
│   └── patch.py         # split, merge
└── fusion/
    ├── __init__.py      # image-mask pair 관련 상위 API 재노출
    ├── otsu.py          # otsu thresholding
    ├── grabcut.py       # grabcut refinement
    ├── guided_filter.py # guided filter refinement
    ├── active_contour.py# active contour refinement
    └── sync.py          # synchronized spatial operation wrappers

tests/
└── test_*.py        # 패키지 구조에 대응하는 기능별 테스트
```

### 5.1 파일별 상세 책임 (Responsibility)
*   **`core.py`**: 입력 데이터의 정규성(np.uint8, [0, 255] for image, [0, 1] for mask)을 검사하고 형식을 강제함.
*   **`image/standard_norm.py`**, **`image/hist_norm.py`**, **`image/clahe.py`**: 정규화/대비 개선 담당.
*   **`image/brightness.py`**, **`image/contrast.py`**: 픽셀 단위 전역 변환 담당.
*   **`image/blur.py`**, **`image/denoise.py`**, **`image/sharpen.py`**: 필터링 및 세부 특징 강화 담당.
*   **`image/__init__.py`**: 이미지 관련 공개 API를 상위에서 정리해 노출.
*   **`mask/morphology.py`**, **`mask/hole.py`**, **`mask/component.py`**: 기본 마스크 정제 담당.
*   **`mask/hull.py`**, **`mask/edge.py`**: 형상 기반 처리 담당.
*   **`mask/distance.py`**: 이미지 형태의 거리 맵 생성 담당.
*   **`mask/helper.py`**: `get_coords`, `get_bbox`, `get_area`, `get_centroid` 같은 요약형 분석 helper 담당.
*   **`mask/__init__.py`**: 마스크 관련 공개 API를 상위에서 정리해 노출.
*   **`spatial/resize.py`**, **`spatial/transform.py`**: 기하학적 변형 담당.
*   **`spatial/pad.py`**, **`spatial/crop.py`**: 여백 및 영역 조정 담당.
*   **`spatial/patch.py`**: patch 분할 및 복원 담당.
*   **`spatial/__init__.py`**: 공간 변환 관련 공개 API를 상위에서 정리해 노출.
*   **`fusion/otsu.py`**: 이미지 기반 마스크 생성 담당.
*   **`fusion/grabcut.py`**, **`fusion/guided_filter.py`**, **`fusion/active_contour.py`**: image-mask refinement 담당.
*   **`fusion/sync.py`**: spatial 연산을 image-mask pair에 맞게 감싸는 synchronized wrapper 담당.
*   **`fusion/__init__.py`**: fusion 관련 공개 API를 상위에서 정리해 노출.

## 6. 향후 명시가 필요한 항목

아래 항목은 구현 전에 추가 합의가 있으면 좋습니다.

*   랜덤성이 있는 알고리즘의 seed 노출 방식
*   `concave hull`, `guided filter`, `active contour`의 의존 라이브러리 기준
*   연산 실패 시 경고와 예외를 구분하는 기준

---
*(이후의 논의 내용은 사용자의 추가 발언에 따라 아래에 계속 업데이트 예정)*
