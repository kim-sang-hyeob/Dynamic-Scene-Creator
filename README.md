# 4DGS Pipeline - Colmap-free Unity→4DGS

**Colmap-free 경량화 및 동적 객체 합성을 위한 4DGS 파이프라인**

> 부스트캠프 AI Tech 8기 CV-09 최종 프로젝트

Unity 카메라 트래킹 데이터를 활용하여 SfM(Structure from Motion) 과정 없이 4D Gaussian Splatting 모델을 학습하는 파이프라인입니다.

## 연구 배경

### 문제 정의
기존 4D Gaussian Splatting(4DGS) 파이프라인은 COLMAP 기반 SfM 과정이 필수적입니다. 이는 다음과 같은 한계가 있습니다:

1. **SfM 실패 문제**: 동적 객체가 포함된 영상에서 feature matching 실패
2. **시간/연산 비용**: COLMAP SfM은 수십 분~수 시간 소요
3. **카메라 자유도 제한**: 학습 시 사용한 카메라 궤적에서만 렌더링 가능

### 해결 방안
Unity 엔진의 **정확한 카메라 트래킹 데이터**를 직접 활용하여:

- **SfM 완전 우회**: Unity JSON → COLMAP format 직접 변환
- **동적 객체 지원**: SfM 없이 동적 씬 학습 가능
- **카메라 자유도 확보**: 학습 후 임의 각도에서 렌더링 (CAMERA_ANGLE_OFFSET)

### 파이프라인 개요
```
Unity Scene → Camera Tracking JSON → Diffusion Video Generation
                    ↓
            process-unity (SfM bypass)
                    ↓
              4DGS Training
                    ↓
         Novel View Rendering (any angle)
```

## Features

- **SfM-free**: Unity의 정확한 카메라 데이터를 직접 활용 (COLMAP 불필요)
- **Camera Rotation Rendering**: 학습된 모델을 다양한 각도에서 렌더링
- **Configurable Coordinate Transform**: Unity↔NeRF 좌표 변환 파라미터 설정 가능
- **V100 GPU Optimized**: CUDA 11.8 + PyTorch 호환성 검증

## Requirements

- Python 3.8+
- CUDA 11.8 (V100 GPU 호환)
- PyTorch (CUDA 11.8 빌드)
- numpy < 2.0

## Installation

### 서버 환경 (V100 GPU)

```bash
# 1. 레포지토리 클론
git clone <your-repo-url>
cd pro-cv-finalproject-cv-09

# 2. 자동 설치 (권장, root로 실행)
chmod +x scripts/setup_server.sh
./scripts/setup_server.sh

# 또는 Python 명령어로 실행
python manage.py setup-server
```

### 수동 설치

```bash
# CUDA 11.8 설치 (V100 필수, root로 실행)
apt-get update
apt-get install -y cuda-nvcc-11-8

# 환경 변수 설정
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# Python 의존성
pip install "numpy<2.0"
pip install websockets

# 4DGS 모델 설치
python manage.py setup --model 4dgs
```

## Quick Start

```bash
# 1. 데이터 처리 (프레임 수 제한 + 이미지 크기 축소로 VRAM 절약)
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat \
    --frames 40 \
    --resize 0.5

# 2. 학습 (--low-vram: batch_size=1로 VRAM 절약)
python manage.py train data/black_cat --low-vram

# 3. 렌더링 (45도 회전)
CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test
```

## Usage

### 1. Unity 데이터 처리

Unity에서 추출한 카메라 트래킹 JSON과 비디오를 4DGS 데이터셋으로 변환합니다.

```bash
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat \
    --frames 40 \
    --resize 0.5
```

**옵션:**
- `--frames 40` - 균일 샘플링으로 40프레임 추출 (첫 프레임과 마지막 프레임 포함)
- `--resize 0.5` - 이미지 크기를 50%로 축소 (VRAM 절약)
- `--resize 384x216` - 또는 특정 해상도로 지정 가능

**입력 파일:**
- `output_cat.mp4` - Diffusion 모델로 생성된 비디오
- `full_data.json` - Unity 카메라 트래킹 데이터
- `original_catvideo.mp4` - 원본 Unity 비디오 (타이밍 동기화용)

**생성 파일:**
- `images/` - 추출된 프레임
- `sync_metadata.json` - 프레임별 Unity 데이터
- `transforms_train.json` - 카메라 매트릭스 (NeRF 포맷)
- `timestamps.json` - 4DGS용 프레임 타임스탬프
- `map_transform.json` - 좌표 변환 파라미터
- `sparse/0/` - COLMAP 호환 포맷

### 2. 4DGS 학습

```bash
python manage.py train data/black_cat
```

### 3. 렌더링 (카메라 회전)

```bash
# 45도 회전된 카메라로 렌더링
CAMERA_ANGLE_OFFSET=45 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test

# 정면 (0도)
CAMERA_ANGLE_OFFSET=0 python external/4dgs/render.py \
    -m output/4dgs/black_cat \
    --skip_train --skip_test
```

### 4. 시각화 (Optional)

```bash
# Rerun에서 시각화
python manage.py visualize output/4dgs/black_cat/point_cloud

# 웹 뷰어로 시각화
python manage.py visualize output/4dgs/black_cat/point_cloud --web
```

## Project Structure

```
4dgs_project/
├── manage.py              # 메인 CLI
├── configs/
│   ├── default.yaml       # 전역 설정
│   └── models/
│       └── 4dgs.yaml      # 4DGS 모델 설정
├── src/
│   ├── setup.py           # 환경 설정 매니저
│   ├── runner.py          # 학습/렌더링 실행기
│   ├── dataset.py         # 데이터셋 매니저
│   ├── model_registry.py  # 모델 레지스트리
│   ├── json_sync_utils.py # Unity JSON 동기화 (핵심)
│   ├── camera_transform.py # 카메라 좌표 변환
│   ├── patch_4dgs_camera_offset.py # 4DGS 카메라 패치
│   ├── filter_utils.py    # PLY 필터링
│   ├── rerun_vis.py       # Rerun 시각화
│   └── exporter.py        # PLY→Splat 변환
├── scripts/
│   └── setup_server.sh    # 서버 자동 설치
├── data/                  # 입력 데이터 (gitignore)
├── external/              # 외부 모델 (gitignore)
│   └── 4dgs/              # 4DGS 레포지토리
├── output/                # 학습 출력 (gitignore)
└── archive/               # 미사용 파일 보관
```

## Coordinate System

Unity와 NeRF/4DGS는 다른 좌표계를 사용합니다. `map_transform.json`으로 변환 파라미터를 설정합니다.

```json
{
    "position": [-150.85, -30.0, 3.66],
    "rotation": [0, 0, 0],
    "scale": [3, 3, 3]
}
```

**변환 공식:**
```
nerf_pos = (unity_pos - map_position) / map_scale
nerf_pos.z = -nerf_pos.z  # Z축 반전
```

### 커스텀 좌표 변환

```bash
# process-unity 시 직접 지정
python manage.py process-unity video.mp4 data.json original.mp4 \
    --output my_scene \
    --map-pos "-150.85,-30.0,3.66" \
    --map-scale "3,3,3"
```

## Camera Rotation

`CAMERA_ANGLE_OFFSET` 환경 변수로 렌더링 시 카메라를 Y축 기준으로 회전합니다.

- **회전 중심**: 객체 위치 (sync_metadata.json의 objPos에서 자동 계산)
- **회전 축**: Y축 (수직)

```bash
# 회전 중심 직접 지정 (선택사항)
CAMERA_ANGLE_OFFSET=45 CAMERA_ROTATION_CENTER="0.5,0.3,-1.2" \
    python external/4dgs/render.py -m output/4dgs/black_cat --skip_train --skip_test
```

## Troubleshooting

### CUDA 버전 문제

V100 GPU는 CUDA 11.8과 호환됩니다. 다른 버전 사용 시 오류 발생 가능:

```bash
# CUDA 버전 확인
nvcc --version

# CUDA_HOME 설정 확인
echo $CUDA_HOME
```

### numpy 버전 문제

numpy 2.0 이상은 호환성 문제가 있습니다:

```bash
pip install "numpy<2.0"
```

### 4DGS 빌드 오류

```bash
# CUDA_HOME이 설정되어 있는지 확인
export CUDA_HOME=/usr/local/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH

# 재설치
python manage.py setup --model 4dgs
```

## Commands Reference

| 명령어 | 설명 |
|--------|------|
| `setup` | 4DGS 환경 설치 |
| `process-unity` | Unity JSON + Video → 4DGS 데이터셋 |
| `train` | 4DGS 모델 학습 |
| `visualize` | Rerun 시각화 |
| `clean-model` | PLY floater 제거 |
| `list-models` | 사용 가능한 모델 목록 |
| `setup-server` | 서버 환경 자동 설치 |

### process-unity 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--output` | 출력 데이터셋 이름 (필수) | `--output black_cat` |
| `--frames` | 균일 샘플링 프레임 수 (첫/끝 포함) | `--frames 40` |
| `--resize` | 이미지 크기 조정 | `--resize 0.5` 또는 `--resize 384x216` |
| `--map-pos` | 좌표 변환 위치 오버라이드 | `--map-pos "-150.85,-30.0,3.66"` |
| `--map-scale` | 좌표 변환 스케일 오버라이드 | `--map-scale "3,3,3"` |

### train 옵션

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--model` | 사용할 모델 (기본: 4dgs) | `--model 4dgs` |
| `--extra` | 추가 학습 인자 | `--extra "--iterations 20000"` |
| `--low-vram` | 저사양 모드 (batch_size=1) | `--low-vram` |

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    Unity→4DGS Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Unity   │    │ Diffusion│    │  4DGS    │    │  Render  │  │
│  │ Tracking │ +  │  Video   │ → │  Train   │ → │  Video   │  │
│  │  JSON    │    │   MP4    │    │          │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       │               │               │               │         │
│       └───────┬───────┘               │               │         │
│               ↓                       │               │         │
│        process-unity              train           render.py     │
│        (SfM bypass!)                              + CAMERA_     │
│                                                   ANGLE_OFFSET  │
└─────────────────────────────────────────────────────────────────┘
```

## Team

**부스트캠프 AI Tech 8기 CV-09**

## License

This project is for educational and research purposes (Boostcamp AI Tech Final Project).

## References

- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
