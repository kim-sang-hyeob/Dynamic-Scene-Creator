# 4DGS Scene Composer

멀티레이어 4D Gaussian Splatting 씬 합성 도구. 여러 3DGS/4DGS 오브젝트를 하나의 씬에 배치하고, TRELLIS AI로 이미지에서 3D 오브젝트를 생성하고, World Labs API로 텍스트/이미지에서 3D Map을 생성하며, 카메라 경로를 편집/녹화할 수 있습니다.

---

## 실행 방법

```bash
# 프론트엔드 서버 (포트 8080)
cd viewer/scene-composer
python3 server.py

# TRELLIS + World Labs 백엔드 서버 (포트 8000, GPU 필요)
cd viewer/trellis-server
ATTN_BACKEND=xformers SPCONV_ALGO=native python3 server.py

# World Labs Map 생성 기능을 사용하려면 API 키 설정:
TRELLIS_WORLDLABS_API_KEY=wl_xxx ATTN_BACKEND=xformers SPCONV_ALGO=native python3 server.py
```

브라우저에서 `http://localhost:8080` 접속

> 포트 충돌 시: `fuser -k 8080/tcp` 또는 `fuser -k 8000/tcp`

---

## 기능 개요

### 1. 멀티레이어 씬 합성

- `.splat`, `.ply`, `.splatv` 파일을 드래그앤드롭으로 추가
- 첫 번째 파일은 **Map** (배경), 이후 파일은 **Object** (조작 가능)
- 레이어별 표시/숨김, 잠금, 삭제
- 레이어별 Position / Rotation / Scale 조정
- CPU Pre-bake 방식: 변환 → CPU에서 위치 재계산 → GPU 단일 업로드

### 2. TRELLIS AI 3D 오브젝트 생성

- 하단 프롬프트 바에서 이미지 업로드 → 3D 오브젝트 자동 생성
- 이미지 드래그앤드롭 또는 `Ctrl+V` 클립보드 붙여넣기
- 생성된 오브젝트는 카메라 전방 3m 위치에 자동 배치
- 크기 자동 정규화 (longest axis = 1.0)

### 3. World Labs Map 생성

- 하단 프롬프트 바의 **Map** 버튼 클릭 → 모달 UI 오픈
- 텍스트 프롬프트 또는 참조 이미지로 3D Map(배경 씬) 생성
- World Labs Marble API 사용 (비동기 폴링 방식)
- SPZ → .splat 자동 변환 (서버사이드)
- 생성된 Map은 자동으로 Map 레이어로 추가

### 4. 직접 조작 (Direct Manipulation)

오브젝트를 마우스로 직접 조작하는 기본 인터랙션:

| 조작 | 동작 |
|------|------|
| **좌클릭 드래그** | 바닥면(XZ) 위에서 이동 |
| **Alt + 좌클릭 드래그** | 높이(Y축) 조절 |
| **우클릭 드래그** | Y축 회전 |
| **마우스 스크롤** | 크기 조절 (위: 확대, 아래: 축소) |
| **Shift** | 그리드 스냅 (0.5 단위) |
| **Arrow Keys** | 미세 이동 (0.1 단위, Shift: 0.5) |
| **PageUp / PageDown** | 높이 미세 조절 |
| **Delete** | 선택된 오브젝트 삭제 |

### 5. 기즈모 (정밀 조작)

축별 정밀 이동/회전이 필요할 때 사용:

| 키 | 기즈모 모드 |
|----|-------------|
| **G** | 이동 기즈모 (Translate) — X/Y/Z 축 + XZ/XY/YZ 평면 핸들 |
| **R** | 회전 기즈모 (Rotate) — X/Y/Z 축별 회전 링 |
| **Esc** | 기즈모 숨김 + 선택 해제 |

- 상단 HUD 버튼 (T / R / S)으로도 전환 가능
- Shift 드래그: 스냅 (이동 0.5, 회전 15도)

### 6. 선택 바운딩 박스

- 오브젝트 선택 시 하늘색 AABB 와이어프레임 표시
- 이동/회전/크기 변경 시 실시간 업데이트

### 7. 카메라 조작

| 조작 | 동작 |
|------|------|
| **빈 곳 좌클릭 드래그** | 카메라 회전 (Orbit) |
| **빈 곳 우클릭 드래그** | 카메라 이동 (Pan) |
| **빈 곳 스크롤** | 줌 인/아웃 |
| **W / A / S / D** | 카메라 전후좌우 이동 |
| **Q / E** | 카메라 상하 이동 |

- 수직 각도 ±30도 제한 (뒤집힘 방지)

### 8. Path Editor (카메라 경로 편집)

사이드바 **Path** 탭에서 카메라 경로를 편집하고 영상을 녹화:

| 모드 | 키 | 설명 |
|------|----|------|
| **View** | 1 | 일반 뷰어 모드 |
| **Place** | 2 | 클릭으로 제어점 배치 |
| **Select** | 3 | 제어점 선택/드래그 이동 |
| **Animate** | 4 | 경로 미리보기 |

- **Natural Cubic Spline** 보간
- **돔 카메라**: 거리, 방위각, 앙각 설정
- **Space**: 애니메이션 재생/정지
- **WebM 녹화**: VP9 코덱 고화질 녹화
- **JSON 내보내기/불러오기**: 경로 데이터 저장
- **Training Dataset Export**: FPS 설정 후 프레임 이미지 내보내기

### 9. Undo / Redo

| 키 | 동작 |
|----|------|
| **Ctrl + Z** | 되돌리기 |
| **Ctrl + Y** / **Ctrl + Shift + Z** | 다시 실행 |

---

## 단축키 전체 목록

| 키 | 기능 |
|----|------|
| **?** | 단축키 도움말 토글 |
| **G** | 이동 기즈모 |
| **R** | 회전 기즈모 |
| **Esc** | 선택 해제 / 도움말 닫기 |
| **Delete** | 선택 레이어 삭제 |
| **Arrow Keys** | 미세 이동 |
| **Ctrl+Z** | Undo |
| **Ctrl+Y** | Redo |
| **Ctrl+V** | 클립보드 이미지 붙여넣기 |
| **W/A/S/D** | 카메라 이동 |
| **Q/E** | 카메라 상하 |
| **1-4** | Path Editor 모드 전환 |
| **Space** | 경로 애니메이션 재생/정지 |

---

## 파일 구조

```
scene-composer/
├── index.html                  # 메인 HTML
├── server.py                   # 개발 서버 + API 프록시 (Python, 포트 8080)
├── css/
│   └── composer.css            # 전체 UI 스타일
└── js/
    ├── main.js                 # 앱 진입점, 드래그앤드롭, 키보드 단축키
    ├── renderer.js             # WebGL2 렌더러 + 소트 워커 + 셰이더
    ├── camera-controls.js      # 카메라 Orbit/Pan/Zoom/WASD/Touch
    ├── scene-manager.js        # 레이어 CRUD, 트랜스폼, 머지
    ├── layer-panel.js          # 사이드바 레이어 패널 UI
    ├── gizmo.js                # 기즈모 조작 로직
    ├── gizmo-geometry.js       # 기즈모 메시 (화살표, 링, 큐브)
    ├── direct-manip.js         # 직접 조작 (드래그, 스크롤, 우클릭)
    ├── selection-box.js        # AABB 와이어프레임 렌더러
    ├── undo-manager.js         # Undo/Redo 스택
    ├── prompt-bar.js           # TRELLIS 프롬프트 바 UI
    ├── worldlabs-modal.js      # World Labs Map 생성 모달 UI
    ├── path-editor-panel.js    # Path Editor 사이드바 패널
    ├── path-editor/
    │   ├── path-editor.js      # 경로 편집 코어 로직
    │   ├── bezier-math.js      # Natural Cubic Spline 수학
    │   └── overlay-renderer.js # 경로/포인트/프러스텀 WebGL 오버레이
    └── utils/
        ├── matrix-math.js      # 행렬 연산 (multiply, invert, rotate, ...)
        ├── splat-loader.js     # .splat/.ply/.splatv 파서
        ├── splat-transform.js  # Gaussian 트랜스폼 베이킹
        └── ray-cast.js         # 레이캐스팅 (피킹, AABB, 기즈모 히트테스트)

trellis-server/
├── server.py                   # FastAPI 백엔드 (TRELLIS + World Labs)
├── config.py                   # 서버 설정 (환경 변수 기반)
├── trellis_wrapper.py          # TRELLIS 모델 래퍼
├── worldlabs_wrapper.py        # World Labs Marble API 래퍼
└── requirements.txt            # Python 의존성
```

---

## 백엔드 API

프론트엔드는 `/api/*` 프록시를 통해 백엔드(포트 8000)에 접근합니다.

### TRELLIS (이미지 → 3D 오브젝트)

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/health` | GET | 서버 상태 확인 |
| `/api/status` | GET | 생성 작업 진행 여부 |
| `/api/generate` | POST | 이미지/텍스트 → 3D Gaussian 생성 |

### World Labs (텍스트/이미지 → 3D Map)

| 엔드포인트 | 메서드 | 설명 |
|------------|--------|------|
| `/api/generate-map` | POST | 텍스트/이미지 → 3D Map 생성 |

**환경 변수:**

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `TRELLIS_WORLDLABS_API_KEY` | World Labs API 키 | (없음 — 설정 시 Map 생성 활성화) |
| `TRELLIS_MOCK` | Mock 모드 (GPU 없이 테스트) | `false` |

---

## 기술 스택

- **렌더링**: WebGL2, 16-bit Counting Sort (Web Worker)
- **4DGS 셰이더**: Temporal motion (HexPlane/MLP 기반)
- **CPU Pre-bake**: 레이어별 원본 데이터 보존 → 트랜스폼 시 CPU에서 재계산
- **TRELLIS**: FastAPI 서버, 이미지 → 3D Gaussian Splatting (서버사이드 PLY→.splat 변환)
- **World Labs**: Marble API, 텍스트/이미지 → 3D Map (비동기 폴링 + SPZ→.splat 변환)
- **API 프록시**: 프론트엔드 서버가 `/api/*` 요청을 백엔드로 프록시 (방화벽 우회)
- **경로 보간**: Natural Cubic Spline (C2 연속)
- **녹화**: MediaRecorder API, VP9 코덱

---

## 지원 포맷

| 포맷 | 확장자 | 설명 |
|------|--------|------|
| Splat | `.splat` | 3DGS 웹 뷰어 표준 포맷 |
| PLY | `.ply` | 3DGS 학습 출력 포맷 |
| SplatV | `.splatv` | 4DGS 애니메이션 포맷 (모션 데이터 포함) |
| SPZ | `.spz` | Niantic 압축 포맷 (World Labs 출력, 서버에서 .splat으로 자동 변환) |

---

## 사이드바 (Layers 탭)

- **레이어 목록**: 각 레이어의 이름, Gaussian 수, 타입 (Map/Object)
- **표시/숨김 토글**: 눈 아이콘 클릭
- **잠금**: 잠금 시 이동/회전/삭제 불가
- **트랜스폼 입력**: Position (X/Y/Z), Rotation (X/Y/Z), Scale
- **+ Add Layer**: 파일 선택 다이얼로그

---

## 주의사항

- WebGL2 필수 (구형 브라우저 미지원)
- TRELLIS 서버는 NVIDIA GPU 필요 (A10 이상 권장)
- TRELLIS 모델 로딩에 약 40초 소요 — 서버 재시작 최소화
- World Labs Map 생성은 API 키 필요 (`TRELLIS_WORLDLABS_API_KEY`)
- 대용량 Gaussian (수백만 이상) 시 브라우저 성능 저하 가능
