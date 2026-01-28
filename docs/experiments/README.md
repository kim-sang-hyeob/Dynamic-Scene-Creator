# 실험 기록: Colmap-Free 4DGS with Background Removal

## 프로젝트 개요

Unity에서 추출한 카메라 데이터와 비디오 확산 모델로 생성한 영상을 사용하여,
COLMAP 없이 4D Gaussian Splatting을 학습하는 파이프라인 개발.

**핵심 도전 과제**: 배경이 제거된 투명 PNG 이미지로 4DGS 학습 시 배경 Gaussian 제거

---

## 실험 목록

| # | 실험명 | 상태 | 핵심 발견 |
|---|--------|------|-----------|
| 01 | [Alpha Channel Handling](./01_alpha_channel_handling.md) | ✅ | 4DGS는 Alpha를 무시함, PNG의 RGB는 Alpha=0이어도 존재 |
| 02 | [Loss Masking](./02_loss_masking.md) | ✅ | Loss Masking 단독으로는 작동 안 함 (초기점 필요) |
| 03 | [Foreground Point Initialization](./03_foreground_point_initialization.md) | ✅ | Alpha mask로 전경 위치에 초기점 생성 |
| 04 | [CUDA Memory Issue](./04_cuda_memory_issue.md) | ✅ | Tensor View → Clone으로 메모리 오류 해결 |

---

## 핵심 성과

### 문제 해결 흐름

```
문제: 투명 PNG로 학습해도 배경 형체가 나타남
         ↓
원인 1: 4DGS가 Alpha 무시 → 실험 01에서 패치
         ↓
원인 2: 배경에도 Loss 발생 → 실험 02에서 Loss Masking
         ↓
원인 3: 초기점이 전경에 없음 → 실험 03에서 해결
         ↓
부수 문제: CUDA crash → 실험 04에서 .clone() 추가
         ↓
최종 결과: 배경 없는 깨끗한 4D Gaussian 학습 성공
```

### 정량적 결과

| 지표 | 기존 (배경 포함) | Loss Masking만 | 최종 (모든 패치) |
|------|-----------------|----------------|-----------------|
| 초기 Points | 1080 | 1080 | 1000 |
| 최종 Points | 168000+ | 1080 (변화없음) | 19000+ |
| PSNR | ~25 | 18.4 (학습 실패) | 33+ |
| 배경 Gaussian | 있음 | 측정 불가 | 없음 |

---

## 기술적 기여

### 1. PNG Alpha Channel의 이해
- Alpha=0이어도 RGB 데이터 존재
- 4DGS 원본 코드의 Alpha 무시 문제 발견

### 2. Loss Masking의 한계 발견
- 초기점이 전경에 없으면 학습 불가 (닭과 달걀 문제)
- Foreground Initialization과 조합 필요

### 3. Camera-guided Point Initialization
- Alpha mask + Camera pose로 전경 3D 점 생성
- Back-projection 기법 활용

### 4. PyTorch 메모리 관리
- Tensor View vs Clone 차이
- GC와 CUDA 메모리 문제 해결

---

## 재현 방법

### 필수 패치 적용

```bash
# 4DGS 설치 후 패치 적용
python src/patch_4dgs_open3d.py external/4dgs
python src/patch_4dgs_sfm_free.py external/4dgs/scene/dataset_readers.py
python src/patch_4dgs_alpha.py external/4dgs
```

### 데이터 준비 (배경 제거 + Foreground 초기화)

```bash
python manage.py process-unity \
    data/video.mp4 \
    data/tracking.json \
    data/original.mp4 \
    --output my_scene \
    --frames 40 \
    --remove-bg  # BiRefNet 배경 제거 + Foreground 초기점 생성
```

### 학습

```bash
python manage.py train data/my_scene --extra="--white_background"
```

---

## 파일 구조

```
docs/experiments/
├── README.md                           # 이 파일
├── 01_alpha_channel_handling.md        # Alpha 채널 문제
├── 02_loss_masking.md                  # Loss Masking
├── 03_foreground_point_initialization.md # 전경 기반 초기화
└── 04_cuda_memory_issue.md             # CUDA 메모리 문제

src/
├── patch_4dgs_alpha.py      # Alpha + Loss Masking 패치
├── patch_4dgs_sfm_free.py   # SfM-free 패치
├── background_remover.py    # BiRefNet 배경 제거
└── json_sync_utils.py       # Foreground 초기점 생성
```

---

## 향후 연구 방향

1. **Multi-view Triangulation**: 더 정확한 초기점 깊이 추정
2. **Adaptive Loss Masking**: 학습 초기에는 masking 약하게, 후반에 강하게
3. **Dynamic Background**: 움직이는 배경 처리

---

## 참고 문헌

- [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians)
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [BiRefNet](https://github.com/ZhengPeng7/BiRefNet)
