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

## 스크립트 사용법

### 1. convert_ply_to_splat.py (PLY → .splat 변환)

3DGS로 학습된 PLY 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

```bash
python convert_ply_to_splat.py <input.ply> -o <output.splat>

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

## 📍 web_path_editor (Camera Path Recorder) 

3DGS 맵 위에서 카메라 경로를 기록하고 영상을 촬영하는 도구입니다.

### 실행

```bash
cd web_path_editor
python server.py
```

브라우저에서 http://localhost:8074 접속 (server.py 에 정의된 포트 사용)

### 조작법

| 조작 | 기능 |
|------|------|
| 마우스 드래그 | 카메라 회전 (Orbit) |
| 우클릭 드래그 / Shift+드래그 | 카메라 이동 (Pan) |
| 마우스 휠 | 줌 인/아웃 |
| **P 키** | 현재 카메라 위치에 웨이포인트 추가 |

### 워크플로우

1. `.splat` 파일을 드래그 앤 드롭하여 맵 로드
2. 마우스로 카메라 위치를 원하는 곳으로 이동
3. **P 키** 또는 📌 버튼을 눌러 웨이포인트 추가
4. 여러 위치에서 반복 (최소 2개 필요)
5. **Start Recording** 버튼 클릭 → 촬영

### 출력 파일

```
output/
├── full_data.json      # 프레임별 카메라 데이터
└── images/
    ├── frame_0000.png
    └── ...
```

### images_to_video.py (이미지 → 동영상 변환)

```bash
cd web_path_editor

# 기본 실행
python images_to_video.py

# 옵션 지정
python images_to_video.py -i ./output/images -o ./output/video.mp4 --fps 30
```

**옵션:**
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `-i, --input` | ./images | 이미지 폴더 경로 |
| `-o, --output` | ./output.mp4 | 출력 동영상 경로 |
| `--fps` | 21 | 프레임 레이트 |
| `--pattern` | frame_*.png | 이미지 파일 패턴 |
| `--use-opencv` | - | FFmpeg 대신 OpenCV 사용 |

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
# 프로젝트 루트로 이동
cd /path/to/pro-cv-finalproject-cv-09-main

# 1. 배경 PLY를 .splat으로 변환
python viewer/convert_ply_to_splat.py background.ply -o viewer/map.splat

# 2. 4DGS 모델을 .splatv로 변환
PYTHONPATH=external/4dgs python viewer/convert_4dgs_to_splatv.py \
    --model_path output/4dgs/racoon \
    --output viewer/model.splatv

# 3. 배경과 객체 병합
python viewer/merge_splat_files.py viewer/map.splat viewer/model.splatv \
    -o viewer/merged.splatv \
    --offset 0 1 0 \
    --scale 0.5

# 4. 웹 뷰어에서 확인
cd viewer/web_viewer && python -m http.server 8080

# 5. 카메라 경로 녹화
cd viewer/web_path_editor && python server.py
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
