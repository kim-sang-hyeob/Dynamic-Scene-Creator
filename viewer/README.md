# Viewer - 4DGS 시각화 및 경로 편집 도구

4D Gaussian Splatting 모델을 웹에서 시각화하고, 카메라 경로를 편집/녹화하기 위한 도구입니다.

## 구성

```
viewer/
├── format_manage.py             # 통합 CLI (convert, merge, list)
├── convert_ply_to_splat.py      # PLY → .splat 변환
├── convert_spz_to_splat.py      # SPZ → .splat 변환 (Niantic 압축 포맷)
├── convert_hexplane_to_splatv.py    # HexPlane 기반 4DGS → .splatv 변환
├── convert_mlp_to_splatv.py         # MLP 기반 4DGS → .splatv 변환
├── merge_splat_files.py         # 3DGS(.splat) + 4DGS(.splatv) 병합
└── web_viewer_final/            # 3DGS 경로 에디터 + 뷰어 + 녹화
    ├── index.html               # UI + 에디터 로직
    ├── hybrid.js                # WebGL Gaussian splat 렌더러
    ├── bezier-math.js           # Natural Cubic Spline 경로 수학
    ├── overlay-renderer.js      # WebGL2 오버레이 (커브, 포인트, 프러스텀)
    └── server.py                # 프레임 이미지 저장 서버 (추후 사용)
```

---

## format_manage.py (통합 CLI)

모든 변환 및 병합 기능을 하나의 CLI로 통합한 도구입니다.

### 명령어 목록

```bash
python format_manage.py --help          # 전체 도움말
python format_manage.py list            # 지원 포맷 목록
python format_manage.py convert --help  # 변환 도움말
python format_manage.py merge --help    # 병합 도움말
```

### convert 명령어

```bash
# PLY → .splat
python format_manage.py convert input.ply -o output.splat

# SPZ → .splat
python format_manage.py convert input.spz -o output.splat

# HexPlane 4DGS → .splatv (4dgs 환경 필요)
python format_manage.py convert --type hexplane \
    --model-path <model_dir> \
    --iteration 14000 \
    --num-samples 20 \
    -o output.splatv

# MLP 4DGS → .splatv (sc4d 환경 필요)
python format_manage.py convert --type mlp \
    --model-dir <s2_dir> \
    --num-samples 30 \
    -o output.splatv
```

### merge 명령어

```bash
# 기본 병합
python format_manage.py merge background.splat object.splatv -o merged.splatv

# 위치/크기 조정
python format_manage.py merge background.splat object.splatv -o merged.splatv \
    --offset 0 1.5 -2 --scale 0.5

# 배경도 조정
python format_manage.py merge background.splat object.splatv -o merged.splatv \
    --bg-offset 0 0 0 --bg-scale 1.0 --bg-rotate 0 90 0
```

### 환경 요구사항

| 변환 타입 | 필요 환경 |
|-----------|-----------|
| PLY/SPZ → splat | 기본 Python (numpy, spz) |
| HexPlane → splatv | 4DGS conda 환경 + PYTHONPATH 설정 |
| MLP → splatv | SC4D conda 환경 (pytorch3d 포함) |
| merge | 기본 Python |

---

## 개별 스크립트 사용법

### 1. convert_spz_to_splat.py (SPZ → .splat 변환)

Niantic의 압축 SPZ 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

> SPZ는 PLY 대비 ~90% 압축률을 제공하는 3DGS 압축 포맷입니다.

```bash
# SPZ 라이브러리 설치 (최초 1회)
git clone https://github.com/nianticlabs/spz.git
cd spz && pip install .

# 단일 파일 변환
python convert_spz_to_splat.py model.spz -o model.splat

# 여러 파일 일괄 변환
python convert_spz_to_splat.py *.spz
```

**옵션:**
| 옵션 | 설명 |
|------|------|
| `input_files` | 입력 SPZ 파일 (필수, 여러 개 가능) |
| `-o, --output` | 출력 파일 경로 (단일 파일 입력 시만 유효) |
| `--slow` | 메모리 효율적 모드 (대용량 파일용) |

**SPZ + 4DGS 병합 워크플로우:**
```bash
# 1. SPZ → .splat 변환
python convert_spz_to_splat.py background.spz -o background.splat

# 2. .splat + .splatv 병합
python merge_splat_files.py background.splat object.splatv -o merged.splatv
```

---

### 3. convert_ply_to_splat.py (PLY → .splat 변환)

3DGS로 학습된 PLY 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

```bash
# 예시: 단일 파일 변환 (정규화 포함)
python convert_ply_to_splat.py point_cloud.ply -o map_normalized.splat \
    --rotate -28.07 0.39 -0.10 \
    --extra-yaw 180

# 예시: 여러 파일 일괄 변환
python convert_ply_to_splat.py *.ply
```

**옵션:**
| 옵션 | 설명 |
|------|------|
| `input_files` | 입력 PLY 파일 (필수, 여러 개 가능) |
| `-o, --output` | 출력 파일 경로 (단일 파일 입력 시만 유효) |
| `--rotate` | 초기 회전 적용 (x y z, 도) - 예: `-28.07 0.39 -0.10` (기울기 보정) |
| `--extra-yaw` | 추가 수직 회전 (도) - 예: `180` (좌우/앞뒤 반전 시 사용) |

**설정 위치:** `web_viewer_final/hybrid.js` 내 `defaultViewMatrix`
```javascript
let defaultViewMatrix = [
  -0.97, 0.13, 0.22, 0,
  0.04, 0.91, -0.41, 0,
  -0.25, -0.39, -0.89, 0,
  -1.32, 1.59, 2.84, 1
];
```

**주의:** 저장되는 카메라 좌표는 **기울어진 좌표계 기준**입니다.
- 다른 시스템(Unity 등)에서 사용 시 좌표 변환 필요

---

### 4. convert_hexplane_to_splatv.py (HexPlane 4DGS → .splatv 변환)

HexPlane 기반 4D Gaussian Splatting 모델을 애니메이션 지원 `.splatv` 포맷으로 변환합니다.

> ⚠️ **환경 요구사항**:
> - 4DGS conda 환경 활성화 필요
> - `PYTHONPATH`에 4DGS 모듈 경로 설정 필요
> - 4DGS 프로젝트 루트에서 실행 권장

```bash
# 4DGS 프로젝트 루트에서 실행
cd <4dgs_project_root>
export PYTHONPATH=.

python convert_hexplane_to_splatv.py \
    --model_path <model_dir> \
    --output output.splatv
```

**필수 옵션:**
| 옵션 | 설명 |
|------|------|
| `--model_path` | 4DGS 학습 결과 디렉토리 (예: `output/4dgs/racoon`) |
| `--output` | 출력 `.splatv` 파일 경로 |

**추가 옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--iteration` | -1 (최신) | 사용할 체크포인트 iteration |
| `--num_samples` | 20 | 모션 샘플링 수 (높을수록 정밀) |

---

### 5. convert_mlp_to_splatv.py (MLP 기반 4DGS → .splatv 변환)

MLP 기반 4DGS 모델(s2 stage)을 `.splatv` 포맷으로 변환합니다.

> ⚠️ **환경 요구사항**:
> - SC4D conda 환경 활성화 필요 (`pytorch3d` 포함)
> - CUDA 및 컴파일러 환경 설정 필요할 수 있음

```bash
# SC4D conda 환경에서 실행
python convert_mlp_to_splatv.py \
    --model_dir <s2_dir> \
    --output output.splatv

# 특정 iteration 사용
python convert_mlp_to_splatv.py \
    --model_dir <s2_dir> \
    --output output.splatv \
    --iteration 8000
```

**필수 옵션:**
| 옵션 | 설명 |
|------|------|
| `--model_dir` | s2 디렉토리 경로 (point_cloud.ply, point_cloud_c.ply, timenet.pth 포함) |
| `--output` | 출력 `.splatv` 파일 경로 |

**추가 옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--iteration` | None (최신) | 특정 iteration 사용 (예: 8000 → point_cloud_8000.ply) |
| `--num_samples` | 30 | 모션 샘플링 수 (높을수록 정밀) |
| `--rotate` | None | 초기 회전 적용 (x y z, 도) - 예: `180 0 0` |

**필요 파일 구조:**
```
s2/
├── point_cloud.ply       # Gaussian 데이터
├── point_cloud_c.ply     # Control points
└── timenet.pth           # MLP 가중치
```

---

### 6. merge_splat_files.py (배경 + 객체 병합)

정적 배경(.splat)과 동적 객체(.splatv)를 하나의 파일로 병합합니다.

```bash
python merge_splat_files.py <background.splat> <object.splatv> -o <output.splatv>

# 기본 병합 (옵션 없음)
python merge_splat_files.py map.splat model.splatv -o merged.splatv

# 자동 바닥 스냅 (lumina_path 필요)
python merge_splat_files.py map.splat model.splatv -o merged.splatv \
    --lumina-path lumina.json --snap-to-floor

### 6.2 Path Animation
Animate an object along a path defined in a JSON file (e.g. from Lumina).
- **Translation Only**: The object slides along the path maintaining its original orientation (no rotation).
- **Absolute Coordinate System**: The object moves exactly along the coordinates defined in the JSON.
- **Constant Speed**: The object moves at a constant speed (units/sec).
- **Play Once & Freeze**: The animation plays from start to end, then freezes at the final position.

```bash
python viewer/merge_splat_files.py viewer/background.splat viewer/object.splatv \
  -o viewer/output.splatv \
  --lumina-path viewer/path.json \
  --animate \
  --speed 2.0
```
# 수동 위치 조정 (필요한 경우에만)
python merge_splat_files.py map.splat model.splatv -o merged.splatv \
    --offset 1.5 0.0 -2.0 --scale 0.5
```

**필수 옵션:**
| 옵션 | 설명 |
|------|------|
| `background` | 배경 .splat 파일 (위치 인수) |
| `object` | 동적 객체 .splatv 파일 (위치 인수) |
| `-o, --output` | 출력 `.splatv` 파일 경로 |

**추가 옵션 (모두 선택 사항):**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--lumina-path` | None | `lumina_path.json` 경로 (바닥 높이 및 경로 참조) |
| `--snap-to-floor` | False | 객체 바닥을 지면에 자동 정렬 (정적 배치용) |
| `--animate` | False | 객체가 경로를 따라 이동하며 회전하도록 애니메이션 베이킹 |
| `--offset X Y Z` | 0 0 0 | 객체 위치 수동 오프셋 |
| `--scale` | 1.0 | 객체 스케일 |
| `--rotate X Y Z` | 0 0 0 | 객체 추가 회전 |
| `--bg-offset X Y Z` | 0 0 0 | 배경 위치 오프셋 |
| `--bg-scale` | 1.0 | 배경 스케일 |
| `--bg-rotate X Y Z` | 0 0 0 | 배경 추가 회전 |

---

## 🎬 web_viewer_final (경로 에디터)

3DGS 맵 위에서 **Natural Cubic Spline** 곡선 경로를 편집하고, 돔 카메라로 경로를 따라가며 WebM 영상을 녹화하는 도구입니다.

### 실행

```bash
cd viewer/web_viewer_final
python3 -m http.server 8090
```

브라우저에서 http://localhost:8090 접속 → `.splat` 파일 드래그앤드롭

### 주요 기능

- **Gaussian Picking**: 화면 클릭 시 가장 가까운 Gaussian의 3D 위치에 제어점 배치
- **Natural Cubic Spline 보간**: 제어점을 C2 연속 곡선으로 자동 연결 (자연 3차 스플라인)
- **돔 카메라 시스템**: 경로의 수평 접선(tangent)을 따라가며 수평 유지
- **지면 자동 감지**: 제어점들의 높이 분포에서 mapUp 방향을 자동 추출
- **WebGL 오버레이**: Gaussian splat 위에 경로 커브 + 제어점 + 카메라 프러스텀 렌더링
- **WebM 녹화**: VP9 코덱, 40Mbps 고화질 녹화 (녹화 중 오버레이 자동 숨김)
- **JSON 내보내기/불러오기**: 경로 데이터 저장 및 재사용

### 에디터 모드

| 모드 | 좌클릭 | 설명 |
|------|--------|------|
| VIEW | 카메라 회전 | 일반 뷰어 모드 |
| PLACE | 제어점 배치 | Gaussian 위치에 클릭으로 포인트 추가 |
| SELECT | 포인트 선택/드래그 | 기존 제어점 이동 |
| ANIMATE | 카메라 회전 | 경로 위 카메라 인디케이터 재생 |

### 조작법

| 조작 | 기능 |
|------|------|
| 좌클릭 드래그 | 카메라 회전 (VIEW/ANIMATE) 또는 포인트 배치/선택 |
| 우클릭 드래그 | 카메라 이동 (Pan) |
| 마우스 휠 | 줌 인/아웃 |
| `W/A/S/D` | 카메라 전후좌우 이동 |
| `1`~`4` | 모드 전환 (VIEW/PLACE/SELECT/ANIMATE) |
| `Delete` | 선택된 포인트 삭제 |
| `Space` | 애니메이션 재생/정지 |

### 돔 카메라 설정

| 옵션 | 설명 |
|------|------|
| Distance | 카메라와 경로 사이 거리 |
| Azimuth | 카메라 수평 회전 각도 (°) |
| Elevation | 카메라 높이 각도 (°) |
| Duration | 애니메이션/녹화 시간 (초) |
| FPS | 초당 프레임 수 |

### 경로 데이터 형식 (JSON)

```json
{
  "controlPoints": [
    { "id": 0, "position": [-1.32, 1.59, 2.84] },
    { "id": 1, "position": [0.50, 1.20, 1.00] }
  ],
  "settings": {
    "tension": 0.5,
    "camDistance": 3,
    "camAzimuth": 0,
    "camElevation": 15,
    "duration": 5,
    "fps": 30
  }
}
```

---

## 워크플로우

> 💡 개별 스크립트 대신 `format_manage.py`를 사용할 수도 있습니다.

### HexPlane 기반 4DGS 사용 시

```bash
# 1. 배경 PLY → .splat 변환 (여기서 회전 보정을 권장!)
python convert_ply_to_splat.py background.ply -o map_normalized.splat \
    --rotate 151.93 0.39 -0.10 \
    --extra-yaw 180

# 2. HexPlane 모델 → .splatv 변환 (PYTHONPATH 설정 필요)
python format_manage.py convert --type hexplane \
    --model-path <model_dir> \
    -o model.splatv

# 3. 배경 + 객체 병합 (보정된 맵 사용 시 옵션 불필요)
python merge_splat_files.py map_normalized.splat model.splatv -o merged.splatv \
    --lumina-path path.json --snap-to-floor

# 4. 경로 에디터 실행
cd web_viewer_final && python3 -m http.server 8090
```

### SPZ 배경 + 4DGS 객체 병합 시

```bash
# 1. SPZ → .splat 변환
python format_manage.py convert background.spz -o map.splat

# 2. HexPlane 모델 → .splatv 변환 (4DGS 환경 필요)
python format_manage.py convert --type hexplane \
    --model-path <model_dir> \
    -o model.splatv

# 3. 배경 + 객체 병합
python format_manage.py merge map.splat model.splatv -o merged.splatv \
    --offset 0 0 0 --scale 1.0

# 4. 경로 에디터에서 확인
cd web_viewer_final && python3 -m http.server 8090
```

### MLP 기반 4DGS 사용 시

```bash
# 1. SC4D 모델 → .splatv 변환 (여기서 회전 보정을 권장!)
conda activate sc4d
python viewer/convert_mlp_to_splatv.py \
    --model_dir SC4D/logs/penguin/s2 \
    --output model_normalized.splatv \
    --rotate 180 0 0

# 2. 경로 에디터에서 확인 (또는 merge)
cd web_viewer_final && python3 -m http.server 8090
```

---

## ⚠️ 주의 사항

### WebGL 좌표계

| 시스템 | 좌표계 |
|--------|--------|
| Unity | 왼손, Y-up |
| Three.js/WebGL | 오른손, Y-up |
| 변환 | (x, y, z) → (x, y, -z) |

### 한계점

- ⚠️ **대용량 파일**: 수백만 가우시안 이상은 브라우저 성능 저하 가능
- ⚠️ **모바일 미지원**: 데스크톱 브라우저 권장
- ⚠️ **WebGL2 필수**: 구형 브라우저 미지원
