# Viewer - 4DGS 시각화 및 경로 편집 도구

4D Gaussian Splatting 모델을 웹에서 시각화하고, 카메라 경로를 편집/녹화하기 위한 도구입니다.

## 구성

```
viewer/
└── web_viewer_final/        # 3DGS 경로 에디터 + 뷰어 + 녹화
    ├── index.html           # UI + 에디터 로직
    ├── hybrid.js            # WebGL Gaussian splat 렌더러
    ├── bezier-math.js       # Natural Cubic Spline 경로 수학
    ├── overlay-renderer.js  # WebGL2 오버레이 (커브, 포인트, 프러스텀)
    └── server.py            # 프레임 이미지 저장 서버 (추후 사용)
```

---

## ⚠️ 중요 사항

### 맵 좌표계 (map.splat)

현재 `map.splat`은 **기울어진 상태**로 저장되어 있습니다.

**처리 방식:**
- ❌ 맵 회전 시도 → 가우시안 품질 저하 발생
- ✅ **카메라 시작 위치만 조정**하여 수평으로 보이게 함

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

### WebGL 좌표계

| 시스템 | 좌표계 |
|--------|--------|
| Unity | 왼손, Y-up |
| Three.js/WebGL | 오른손, Y-up |
| 변환 | (x, y, z) → (x, y, -z) |

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
# 경로 에디터 실행
cd viewer/web_viewer_final
python3 -m http.server 8090

# 1. .splat 파일 드래그앤드롭으로 맵 로드
# 2. PLACE 모드에서 클릭으로 경로 제어점 배치
# 3. 돔 카메라 설정 (Distance, Elevation 등) 조정
# 4. ANIMATE 모드에서 경로 미리보기
# 5. Record 버튼으로 WebM 녹화
# 6. Save Path로 경로 JSON 내보내기
```
