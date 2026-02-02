# Viewer - 4DGS 시각화 및 경로 편집 도구

4D Gaussian Splatting 모델을 웹에서 시각화하고, 카메라 경로를 편집/녹화하기 위한 도구입니다.

## 구성

```
viewer/
├── convert_ply_to_splat.py      # PLY → .splat 변환
├── convert_4dgs_to_splatv.py    # 4DGS 학습 결과 → .splatv 변환
├── merge_splat_files.py         # 3DGS(.splat) + 4DGS(.splatv) 병합
└── web_viewer_final/            # 3DGS 경로 에디터 + 뷰어 + 녹화
    ├── index.html               # UI + 에디터 로직
    ├── hybrid.js                # WebGL Gaussian splat 렌더러
    ├── bezier-math.js           # Natural Cubic Spline 경로 수학
    ├── overlay-renderer.js      # WebGL2 오버레이 (커브, 포인트, 프러스텀)
    └── server.py                # 프레임 이미지 저장 서버 (추후 사용)
```

---

## 스크립트 사용법

### 1. convert_ply_to_splat.py (PLY → .splat 변환)

3DGS로 학습된 PLY 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

```bash
# 예시: 단일 파일 변환
python convert_ply_to_splat.py point_cloud.ply -o map.splat

# 예시: 여러 파일 일괄 변환 (각각 .splat 파일 생성)
python convert_ply_to_splat.py *.ply
```

**옵션:**
| 옵션 | 설명 |
|------|------|
| `input_files` | 입력 PLY 파일 (필수, 여러 개 가능) |
| `-o, --output` | 출력 파일 경로 (단일 파일 입력 시만 유효) |

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

### 2. convert_4dgs_to_splatv.py (4DGS → .splatv 변환)

4D Gaussian Splatting 모델을 애니메이션 지원 `.splatv` 포맷으로 변환합니다.

> ⚠️ **주의**: 이 스크립트는 4DGS 모듈을 사용하므로 `PYTHONPATH` 설정과 프로젝트 루트에서 실행이 필요합니다.

```bash
# 프로젝트 루트에서 실행
cd /path/to/pro-cv-finalproject-cv-09-main

PYTHONPATH=external/4dgs python viewer/convert_4dgs_to_splatv.py \
    --model_path output/4dgs/<dataset_name> \
    --output viewer/model.splatv
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

### 3. merge_splat_files.py (배경 + 객체 병합)

정적 배경(.splat)과 동적 객체(.splatv)를 하나의 파일로 병합합니다.

```bash
python merge_splat_files.py <background.splat> <object.splatv> -o <output.splatv>

# 기본 병합
python merge_splat_files.py map.splat model.splatv -o merged.splatv

# 객체 위치/크기 조정
python merge_splat_files.py map.splat model.splatv -o merged.splatv \
    --offset 1.5 0.0 -2.0 \
    --scale 0.5
```

**필수 옵션:**
| 옵션 | 설명 |
|------|------|
| `background` | 배경 .splat 파일 (위치 인수) |
| `object` | 동적 객체 .splatv 파일 (위치 인수) |
| `-o, --output` | 출력 `.splatv` 파일 경로 |

**추가 옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--offset X Y Z` | 0 0 0 | 객체 위치 오프셋 |
| `--scale` | 1.0 | 객체 스케일 |
| `--bg-offset X Y Z` | 0 0 0 | 배경 위치 오프셋 |
| `--bg-scale` | 1.0 | 배경 스케일 |
| `--bg-rotate X Y Z` | 0 0 0 | 배경 회전 (도) |

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

```bash
# 1. 배경 PLY → .splat 변환
python convert_ply_to_splat.py background.ply -o map.splat

# 2. 4DGS 모델 → .splatv 변환
python convert_4dgs_to_splatv.py ./output/point_cloud/iteration_30000 -o model.splatv

# 3. 배경 + 객체 병합
python merge_splat_files.py map.splat model.splatv -o merged.splatv

# 4. 경로 에디터 실행
cd web_viewer_final && python3 -m http.server 8090
# → .splat 드래그앤드롭 → 경로 편집 → WebM 녹화
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
