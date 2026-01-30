# Viewer - 4DGS 시각화 도구

4D Gaussian Splatting 모델을 웹에서 시각화하기 위한 변환 스크립트와 웹 뷰어 모음입니다.

## 📁 구성

| 파일 | 설명 |
|------|------|
| `convert_ply_to_splat.py` | PLY → .splat 변환 |
| `convert_4dgs_to_splatv.py` | 4DGS → .splatv 변환 |
| `merge_splat_files.py` | .splat + .splatv 병합 |
| `web_viewer/` | 웹 기반 뷰어 |

---

## 🔧 스크립트 사용법

### 1. PLY → .splat 변환

3DGS로 학습된 PLY 파일을 웹 뷰어용 `.splat` 포맷으로 변환합니다.

```bash
python convert_ply_to_splat.py <input.ply> -o <output.splat>

# 예시
python convert_ply_to_splat.py point_cloud.ply -o map.splat
```

**옵션:**
- `--sh-mode {first,average}`: SH 계수 처리 방식 (기본: first)

---

### 2. 4DGS → .splatv 변환

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

### 3. 배경 + 객체 병합

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

## 🌐 웹 뷰어

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

## 📋 워크플로우 예시

```bash
# 1. 배경 PLY를 .splat으로 변환
python convert_ply_to_splat.py background.ply -o map.splat

# 2. 4DGS 모델을 .splatv로 변환
python convert_4dgs_to_splatv.py ./4dgs_output/point_cloud/iteration_30000 -o model.splatv

# 3. 배경과 객체 병합
python merge_splat_files.py map.splat model.splatv -o merged.splatv --offset 0 1 0 --scale 0.5

# 4. 웹 뷰어에서 확인
cd web_viewer && python -m http.server 8080
```
