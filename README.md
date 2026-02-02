# 4DGS Pipeline - Colmap-free Unity→4DGS

**고품질 정적 3DGS 맵 위에 동적 객체를 분리 학습하는 Colmap-free 4DGS 파이프라인**

> 부스트캠프 AI Tech 8기 CV-09 최종 프로젝트

Unity 카메라 트래킹 데이터를 활용하여 SfM(Structure from Motion) 과정 없이 4D Gaussian Splatting 모델을 학습하는 파이프라인입니다.

---

## 차별화 포인트

### 1. Large Translational Motion

기존 4DGS 연구는 **제자리에서 움직이는 객체**(손 흔들기, 표정 변화 등)에 최적화되어 있습니다.
우리는 **공간을 가로질러 이동하는 객체**(뛰어가는 고양이, 걸어가는 사람 등)를 타겟으로 합니다.

| | 기존 4DGS | 우리 파이프라인 |
|---|-----------|---------------|
| 대상 모션 | 제자리 움직임 (quasi-static) | 큰 이동 (large translational) |
| Deformation | 작은 변위 보정 | 큰 공간 이동 + 형태 변형 |
| 초기 포인트 | SfM 포인트 사용 | MiDaS 깊이 기반 전경 초기화 |

> SPIN-4DGS, PMGS 등 최신 연구에서도 large translational motion은 **열린 문제**로 언급됨

### 2. Static-Dynamic Composition

기존 접근법은 전체 씬을 하나의 4DGS로 학습합니다.
우리는 **정적 배경과 동적 객체를 분리**하는 전략을 사용합니다.

```
기존: 전체 씬 → 하나의 4DGS → 배경+객체 혼합
우리: 정적 배경 → 고품질 3DGS 맵 (재사용 가능)
      동적 객체 → 배경 제거 후 4DGS 학습 (교체/편집 가능)
```

**장점:**
- 동적 객체만 교체/편집 가능 (씬 전체 재학습 불필요)
- 기존 고품질 3DGS 맵 재활용
- 배경 Gaussian 제거로 학습 효율 향상 (PSNR 33+ 달성)

### 3. Colmap-free Pipeline

합성 환경(Unity)에서 카메라 GT를 직접 획득하여 COLMAP SfM 과정을 완전히 우회합니다.

| | COLMAP 기반 | 우리 파이프라인 |
|---|------------|---------------|
| 카메라 포즈 | SfM으로 추정 (실패 가능) | Unity GT 직접 사용 |
| 초기 포인트 | SfM 포인트 | Alpha mask + MiDaS 깊이 back-projection |
| 전처리 시간 | 수십 분~수 시간 | 수 분 이내 |
| 동적 씬 | SfM 실패 위험 | 문제 없음 |

---

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
              ├─ BiRefNet 배경 제거
              ├─ MiDaS 깊이 추정 → 전경 포인트 초기화
              └─ Unity→COLMAP/NeRF 좌표 변환
                    ↓
              4DGS Training (배경 제거 + Loss Masking)
                    ↓
         Novel View Rendering (any angle)
```

## Features

- **SfM-free**: Unity의 정확한 카메라 데이터를 직접 활용 (COLMAP 불필요)
- **배경 제거 학습**: BiRefNet 배경 제거 (`--remove-bg`) + Alpha-aware Loss Masking
- **MiDaS 깊이 초기화** (선택적): 단안 깊이 추정으로 전경 포인트 클라우드 생성 (`--no-midas`로 비활성화 가능)
- **Camera Rotation Rendering**: 학습된 모델을 다양한 각도에서 렌더링
- **Configurable Coordinate Transform**: Unity↔NeRF 좌표 변환 파라미터 설정 가능
- **Trajectory Visualization**: Rerun 기반 Gaussian 궤적 시각화

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
# 기본 사용법
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat \
    --frames 40 \
    --resize 0.5

# 배경 제거 포함 (BiRefNet 사용)
python manage.py process-unity \
    data/black_cat/output_cat.mp4 \
    data/black_cat/full_data.json \
    data/black_cat/original_catvideo.mp4 \
    --output black_cat_alpha \
    --frames 40 \
    --resize 0.5 \
    --remove-bg
```

**옵션:**
- `--frames 40` - 균일 샘플링으로 40프레임 추출 (첫 프레임과 마지막 프레임 포함)
- `--resize 0.5` - 이미지 크기를 50%로 축소 (VRAM 절약)
- `--resize 384x216` - 또는 특정 해상도로 지정 가능
- `--remove-bg` - BiRefNet으로 배경 제거 (투명 PNG 생성, 학습 시 `--white_background` 필요)
- `--no-midas` - MiDaS 깊이 추정 비활성화 (`--remove-bg` 사용 시 기본은 MiDaS 활성화)

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

### 2. 배경 제거 파이프라인 (Unity 없이)

Unity 카메라 데이터 없이 비디오만으로 4DGS 학습을 위한 데이터셋을 생성합니다.
BiRefNet을 사용하여 배경을 자동으로 제거하고, 고정 카메라 포즈로 COLMAP sparse 파일을 생성합니다.

```bash
# 전체 파이프라인 (배경 제거 + sparse 생성)
python manage.py prepare-alpha data/my_video.mp4 \
    --output my_scene_alpha \
    --frames 40 \
    --resize 512x295

# 학습 (투명 배경용 --white_background 필수)
python manage.py train data/my_scene_alpha --extra="--white_background"
```

**옵션:**
- `--frames 40` - 균일 샘플링으로 40프레임 추출
- `--resize 0.5` - 이미지 크기 50% 축소 (또는 `512x295` 형태)
- `--fov 50` - 카메라 FOV 설정 (기본: 50도)
- `--model birefnet` - 배경 제거 모델 (birefnet/rembg)

**생성 파일:**
- `images/` - 배경이 제거된 투명 PNG 프레임
- `sparse/0/` - COLMAP 호환 포맷 (고정 카메라)
- `timestamps.json` - 4DGS용 프레임 타임스탬프

**개별 명령어:**
```bash
# 1. 배경 제거만
python manage.py remove-bg data/my_video.mp4 --output data/my_scene/images

# 2. sparse 파일 생성만
python manage.py create-sparse data/my_scene/images --fov 50
```

### 3. 4DGS 학습

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

## 기술적 기여 및 실험 결과

### 핵심 문제 해결 흐름

```
문제: 투명 PNG로 학습해도 배경 형체가 나타남
         ↓
원인 1: 4DGS가 Alpha 무시 → Alpha Patch 적용
         ↓
원인 2: 배경에도 Loss 발생 → Loss Masking Patch
         ↓
원인 3: 초기점이 전경에 없음 → MiDaS 깊이 기반 전경 초기화
         ↓
부수 문제: CUDA crash → Tensor View→Clone 수정
         ↓
최종 결과: 배경 없는 깨끗한 4D Gaussian 학습 성공
```

### 정량적 결과

| 지표 | 기존 (배경 포함) | Loss Masking만 | 최종 (모든 패치) |
|------|-----------------|----------------|-----------------|
| 초기 Points | 1080 | 1080 | 1000 |
| 최종 Points | 168000+ | 1080 (변화없음) | 19000+ |
| PSNR | ~25 | 18.4 (학습 실패) | **33+** |
| 배경 Gaussian | 있음 | 측정 불가 | **없음** |

> 자세한 실험 기록은 [docs/experiments/](docs/experiments/) 참고

## Project Structure

```
4dgs_project/
├── manage.py                  # 메인 CLI
├── configs/
│   ├── default.yaml           # 전역 설정
│   └── models/
│       └── 4dgs.yaml          # 4DGS 모델 설정
├── src/
│   ├── converters/            # ★ Unity → 4DGS 데이터 변환 (핵심)
│   │   ├── frame_extractor.py     # 비디오 프레임 추출 + JSON 동기화
│   │   ├── coordinate.py          # Unity ↔ NeRF 좌표계 변환
│   │   ├── colmap_writer.py       # COLMAP sparse 포맷 생성
│   │   ├── nerf_writer.py         # transforms_train.json 생성
│   │   └── sparse_from_images.py  # 이미지 → COLMAP sparse (Unity 없이)
│   ├── adapters/              # 외부 모델/도구 래퍼
│   │   ├── background_remover.py  # BiRefNet 배경 제거
│   │   ├── depth_estimator.py     # MiDaS 깊이 추정
│   │   ├── camera_transform.py    # 렌더링용 카메라 변환
│   │   ├── rerun_vis.py           # Rerun 시각화
│   │   └── visualize_trajectory.py # Gaussian 궤적 시각화
│   ├── patches_4dgs/          # 4DGS 패치 (setup 시 적용)
│   │   ├── alpha.py           # Alpha + Loss Masking 패치
│   │   ├── sfm_free.py        # SfM-free 동작 패치
│   │   ├── open3d.py          # open3d 의존성 제거 패치
│   │   └── camera_offset.py   # 카메라 회전 패치
│   ├── utils/                 # 유틸리티
│   │   ├── filter.py          # PLY floater 제거
│   │   └── exporter.py        # PLY → splat 변환
│   ├── runner.py              # 학습/렌더링 실행기
│   ├── setup.py               # 환경 설정 매니저
│   ├── dataset.py             # 데이터셋 매니저
│   └── model_registry.py      # 모델 레지스트리
├── docs/experiments/          # 실험 기록
├── scripts/
│   └── setup_server.sh        # 서버 자동 설치
├── inputs/                    # 원본 입력 파일 (폴더만 git 추적)
├── data/                      # 변환된 데이터셋 (gitignore)
├── external/4dgs/             # 4DGS 레포지토리 (gitignore)
└── output/                    # 학습 출력 (gitignore)
```

## Data Flow

원본 데이터에서 학습된 모델까지의 흐름입니다.

```
inputs/                          # 1. 원본 입력 파일
├── black_cat/
│   ├── output_cat.mp4           # Diffusion 생성 비디오
│   ├── full_data.json           # Unity 카메라 트래킹
│   └── original.mp4             # 원본 Unity 비디오
│
│   python manage.py process-unity inputs/black_cat/output_cat.mp4 \
│       inputs/black_cat/full_data.json inputs/black_cat/original.mp4 \
│       --output black_cat --frames 40 --resize 0.5
│                    ↓ converters/
│
data/                            # 2. 변환된 4DGS 데이터셋
├── black_cat/
│   ├── images/                  # 추출된 프레임
│   ├── transforms_train.json    # 카메라 매트릭스 (NeRF 포맷)
│   ├── timestamps.json          # 프레임 타임스탬프
│   ├── sparse/0/                # COLMAP 호환 포맷
│   └── map_transform.json       # 좌표 변환 파라미터
│
│   python manage.py train data/black_cat
│                    ↓ train
│
output/                          # 3. 학습된 모델
└── 4dgs/
    └── black_cat/
        └── point_cloud/
            └── iteration_30000/
                ├── point_cloud.ply
                └── deformation.pth
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

## Technical Notes

### PNG 투명도와 4DGS의 Alpha 처리

투명 PNG 이미지를 4DGS에서 학습할 때 알아야 할 중요한 사항입니다.

#### PNG Alpha Channel의 구조

PNG 이미지는 4개 채널로 구성됩니다:
```
픽셀 = [R, G, B, A]
        ↑        ↑
      색상값   투명도 (0=투명, 1=불투명)
```

**핵심 포인트**: Alpha=0 (투명)이어도 R, G, B 값은 **그대로 저장**됩니다.

```python
# 예: BiRefNet으로 배경 제거 후
# 배경 픽셀: R=128, G=130, B=125, A=0
# 이미지 뷰어에서는 안 보이지만 RGB 데이터는 존재!
```

#### 원본 4DGS의 문제점

원본 4DGS(hustvl/4DGaussians)는 alpha 채널을 **무시**합니다:

```python
# camera_utils.py (원본)
return Camera(..., gt_alpha_mask=None, ...)  # Alpha 무시!

# cameras.py (원본)
self.original_image = image[:3,:,:]  # RGB만 사용, Alpha 버림
```

결과: 투명 PNG를 사용해도 배경의 RGB 값이 그대로 학습됨 → 배경 형체가 point cloud에 나타남

#### 우리의 해결책: Alpha Patch (`src/patches_4dgs/alpha.py`)

이 패치는 다음 기능을 포함합니다:
- Alpha 채널 추출
- GT 이미지를 흰색 배경에 합성 (--white_background와 매칭)
- Loss 계산에서 배경 픽셀 제외 (Loss Masking)
- 배경에 Gaussian이 형성되지 않음

```python
# cameras.py (패치 후) - 흰색 배경 합성
if gt_alpha_mask is not None:
    # RGB * alpha + white * (1 - alpha)
    self.original_image = self.original_image * gt_alpha_mask + (1.0 - gt_alpha_mask)

# train.py (패치 후) - Loss Masking
mask_binary = (combined_mask > 0.5).float()
Ll1 = torch.abs(render * mask - gt * mask).sum() / mask.sum()
```

| 영역 | Alpha | Loss 계산 | Gaussian 형성 |
|------|-------|-----------|---------------|
| 전경 (객체) | 1.0 | O | O |
| 배경 | 0.0 | X | X |

#### 사용법

```bash
# 배경 제거된 데이터 학습
python manage.py train data/my_scene_alpha --low-vram --extra="--white_background"
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
| `prepare-alpha` | 비디오 → 배경 제거 + sparse 생성 (Unity 없이) |
| `remove-bg` | 비디오에서 배경 제거 (BiRefNet) |
| `create-sparse` | 이미지 폴더 → COLMAP sparse 파일 |
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
| `--remove-bg` | BiRefNet으로 배경 제거 | `--remove-bg` |
| `--no-midas` | MiDaS 깊이 추정 비활성화 | `--remove-bg --no-midas` |

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

- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians) - 동적 씬 재구성
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - 정적 씬 재구성
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet) - 배경 제거
- [MiDaS](https://github.com/isl-org/MiDaS) - 단안 깊이 추정
- [Rerun](https://github.com/rerun-io/rerun) - 3D 시각화
