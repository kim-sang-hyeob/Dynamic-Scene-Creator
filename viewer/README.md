# Viewer - 4DGS 시각화 도구

4D Gaussian Splatting 모델을 웹에서 시각화하기 위한 변환 스크립트와 웹 뷰어 모음입니다.

## 구성

| 폴더/파일 | 설명 |
|------|------|
| `convert_ply_to_splat.py` | PLY → .splat 변환 |
| `convert_4dgs_to_splatv.py` | 4DGS → .splatv 변환 |
| `merge_splat_files.py` | .splat + .splatv 병합 |
| `web_viewer/` | 웹 기반 뷰어 |
| `web_path_editor/` | 카메라 경로 레코더 |

---

## ⚠️ 중요 사항

### 맵 좌표계 (map.splat)

현재 `map.splat`은 **기울어진 상태**로 저장되어 있습니다.

**처리 방식:**
- ❌ 맵 회전 시도 → 가우시안 품질 저하 발생
- ✅ **카메라 시작 위치만 조정**하여 수평으로 보이게 함

**설정 위치:** `web_path_editor/hybrid.js` (460번줄)
```javascript
// 기울어진 맵에 맞춘 카메라 위치
let defaultViewMatrix = [
  -0.97, 0.13, 0.22, 0,
  0.04, 0.91, -0.41, 0,
  -0.25, -0.39, -0.89, 0,
  -1.32, 1.59, 2.84, 1
];
```

**주의:** 저장되는 카메라 좌표(`full_data.json`)는 **기울어진 좌표계 기준**입니다.
- 다른 시스템(Unity 등)에서 사용 시 좌표 변환 필요
- 변환 공식: 카메라 회전 (153°, -14°, -176°)의 역행렬 적용

### WebGL 좌표계

| 시스템 | 좌표계 |
|--------|--------|
| Unity | 왼손, Y-up |
| Three.js/WebGL | 오른손, Y-up |
| 변환 | (x, y, z) → (x, y, -z) |

---

## 📍 web_path_editor (Camera Path Recorder) 

3DGS 맵 위에서 카메라 경로를 기록하고 영상을 촬영하는 도구입니다.

### 실행

```bash
cd web_path_editor
python server.py
```

브라우저에서 http://localhost:8075 접속

### 워크플로우

1. `.splat` 파일을 드래그 앤 드롭하여 맵 로드
2. 마우스로 카메라 위치를 원하는 곳으로 이동
3. **P 키** 또는 📌 버튼을 눌러 웨이포인트 추가
4. 여러 위치에서 반복하여 경로 생성 (최소 2개 필요)
5. **Start Recording** 버튼 클릭 → 카메라가 경로를 따라 이동하며 촬영

### 조작법

| 조작 | 기능 |
|------|------|
| 마우스 드래그 | 카메라 회전 (Orbit) |
| 우클릭 드래그 / Shift+드래그 | 카메라 이동 (Pan) |
| 마우스 휠 | 줌 인/아웃 |
| **P 키** | 현재 카메라 위치에 웨이포인트 추가 |

### 설정

| 옵션 | 설명 |
|------|------|
| Duration | 전체 녹화 시간 (초) |
| FPS | 초당 프레임 수 |
| Total Frames | Duration × FPS |

### 출력 파일

```
output/
├── full_data.json      # 프레임별 카메라 데이터
└── images/
    ├── frame_0000.png
    ├── frame_0001.png
    └── ...
```

**full_data.json 구조:**
```json
{
  "frames": [
    {
      "frame_idx": 0,
      "time": 0,
      "cam_pos": { "x": -2.05, "y": 1.50, "z": 3.34 },
      "cam_rot": { "x": 113.45, "y": 61.77, "z": 122.63 },
      "cam_fov": 60,
      "viewMatrix": [16개 값...]
    },
    ...
  ]
}
```

### 촬영 원리

- **선형 보간(Linear Interpolation)**: 웨이포인트 간 viewMatrix를 선형 보간
- **일정 속도**: Duration 내에서 모든 구간을 균등 분배
- 핀이 3개면 2개 구간, Duration 6초면 각 구간 3초

---

## 스크립트 사용법

### 1. convert_ply_to_splat.py (PLY → .splat 변환)

3DGS로 학습된 PLY 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

```bash
python convert_ply_to_splat.py <input.ply> -o <output.splat>

# 예시
python convert_ply_to_splat.py point_cloud.ply -o map.splat
```

**옵션:**
- `--sh-mode {first,average}`: SH 계수 처리 방식 (기본: first)

---

### 2. convert_4dgs_to_splatv.py (4DGS → .splatv 변환)

4D Gaussian Splatting 모델을 애니메이션 지원 `.splatv` 포맷으로 변환합니다.

```bash
python convert_4dgs_to_splatv.py <point_cloud_dir> -o <output.splatv>

# 예시 (4DGaussians 학습 결과)
python convert_4dgs_to_splatv.py output/lego/point_cloud/iteration_30000 -o model.splatv
```

**옵션:**
- `--cameras <path>`: 카메라 정보 JSON 파일
- `--num-samples <N>`: 모션 샘플 수 (기본: 20)

---

### 3. merge_splat_files.py (배경 + 객체 병합)

정적 배경(.splat)과 동적 객체(.splatv)를 하나의 파일로 병합합니다.

```bash
python merge_splat_files.py <background> <object> -o <output.splatv>

# 기본 병합
python merge_splat_files.py map.splat model.splatv -o merged.splatv

# 객체 위치/크기 조정
python merge_splat_files.py map.splat model.splatv -o merged.splatv \
    --offset 1.5 0.0 -2.0 \
    --scale 0.5
```

**옵션:**
- `--offset X Y Z`: 객체 위치 오프셋
- `--scale S`: 객체 스케일
- `--bg-offset X Y Z`: 배경 위치 오프셋
- `--bg-scale S`: 배경 스케일

---

## web_viewer (웹 뷰어)

### 실행

```bash
cd web_viewer
python -m http.server 8080
```

브라우저에서 http://localhost:8080 접속

### 파일 로드

`.ply`, `.splat`, `.splatv` 파일을 브라우저 창에 드래그 앤 드롭

### 조작법

| 조작 | 기능 |
|------|------|
| 왼쪽 드래그 | 카메라 회전 (Orbit) |
| 오른쪽 드래그 / Shift+드래그 | 카메라 이동 (Pan) |
| 마우스 휠 | 줌 인/아웃 |
| M 키 | 현재 위치 좌표 복사 |
| V 키 | 뷰 매트릭스 URL에 저장 |

---

## 워크플로우 예시

```bash
# 1. 배경 PLY를 .splat으로 변환
python convert_ply_to_splat.py background.ply -o map.splat

# 2. 4DGS 모델을 .splatv로 변환
python convert_4dgs_to_splatv.py ./4dgs_output/point_cloud/iteration_30000 -o model.splatv

# 3. 배경과 객체 병합
python merge_splat_files.py map.splat model.splatv -o merged.splatv --offset 0 1 0 --scale 0.5

# 4. 웹 뷰어에서 확인
cd web_viewer && python -m http.server 8080

# 5. 카메라 경로 녹화
cd web_path_editor && python server.py
```

---

## 개선점 (Improvements)

### Camera Path Recorder
- ✅ 웨이포인트 기반 직관적인 경로 설정
- ✅ P 키로 현재 카메라 위치에 즉시 핀 추가
- ✅ 실시간 카메라 위치/회전 표시
- ✅ Duration/FPS 슬라이더로 간편한 설정
- ✅ 프레임별 이미지 + JSON 데이터 동시 저장
- ✅ viewMatrix 저장으로 정확한 시점 재현 가능

### 웹 뷰어
- ✅ PLY/Splat/Splatv 파일 드래그 앤 드롭 지원
- ✅ 마우스 기반 직관적인 카메라 조작
- ✅ 위치 정보 클립보드 복사 (M 키)

---

## 한계점 (Limitations)

### Camera Path Recorder
- ⚠️ **좌표계 비정렬**: 맵이 기울어진 경우 저장되는 좌표도 기울어진 좌표계 기준
  - 해결책: 외부에서 좌표 변환 적용 필요
- ⚠️ **선형 보간만 지원**: 곡선 경로(Bezier) 미지원
- ⚠️ **일정 속도만**: 가속/감속(Easing) 미지원
- ⚠️ **단일 객체**: 박스/객체 경로 녹화 기능은 제거됨 (웨이포인트로 대체)

### 웹 뷰어
- ⚠️ **대용량 파일**: 수백만 가우시안 이상은 브라우저 성능 저하 가능
- ⚠️ **모바일 미지원**: 데스크톱 브라우저 권장
- ⚠️ **WebGL2 필수**: 구형 브라우저 미지원

### 파일 변환
- ⚠️ **정밀도 손실**: PLY → Splat 변환 시 SH 계수 압축으로 미세한 색상 차이 발생 가능
- ⚠️ **가우시안 방향 회전**: 맵 전체 회전 시 개별 가우시안 방향도 함께 회전해야 함

---

## 향후 개선 가능 사항

- [ ] Bezier 곡선 경로 지원
- [ ] 가속/감속 Easing 함수 추가
- [ ] 좌표계 변환 옵션 (Unity ↔ WebGL)
- [ ] 박스/객체 경로와 카메라 경로 분리 녹화

