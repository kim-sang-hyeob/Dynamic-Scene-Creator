# 실험 01: 4DGS의 Alpha Channel 처리 문제

## 실험 정보
- **날짜**: 2025-01-28
- **관련 파일**: `patch_4dgs_alpha.py`, `background_remover.py`
- **상태**: ✅ 해결됨

---

## 1. 문제 발견

### 현상
BiRefNet으로 배경을 제거한 투명 PNG 이미지로 4DGS를 학습했으나, 렌더링 결과에 **배경 형체가 그대로 나타남**.

### 기대 결과
```
입력: 고양이만 있는 투명 PNG
출력: 고양이만 있는 3D Gaussian
```

### 실제 결과
```
입력: 고양이만 있는 투명 PNG
출력: 고양이 + 배경 형체가 있는 3D Gaussian
```

---

## 2. 원인 분석

### 가설 1: PNG Alpha 채널이 제대로 저장되지 않음
**검증**: Python으로 PNG 파일 확인
```python
from PIL import Image
import numpy as np
img = np.array(Image.open('image.png'))
print(img.shape)  # (H, W, 4) - RGBA 정상
print(img[:,:,3].min(), img[:,:,3].max())  # 0, 255 - Alpha 정상
```
**결과**: ❌ Alpha 채널은 정상적으로 저장됨

### 가설 2: 4DGS가 Alpha 채널을 무시함
**검증**: 4DGS 소스코드 분석

```python
# external/4dgs/utils/camera_utils.py (원본)
def loadCam(args, id, cam_info, resolution_scale):
    return Camera(..., gt_alpha_mask=None, ...)  # Alpha 무시!
```

```python
# external/4dgs/scene/cameras.py (원본)
self.original_image = image[:3,:,:]  # RGB만 사용, Alpha 버림
```

**결과**: ✅ **4DGS는 Alpha 채널을 완전히 무시함**

### 핵심 발견: PNG Alpha의 구조

```
PNG 픽셀 = [R, G, B, A]
            ↑        ↑
          색상값   투명도 (0=투명, 255=불투명)

중요: Alpha=0 (투명)이어도 R, G, B 값은 그대로 저장됨!
```

BiRefNet 배경 제거 후:
```
배경 픽셀: R=128, G=130, B=125, A=0
         ↑ 색상 데이터 존재    ↑ 투명
```

이미지 뷰어에서는 투명하게 보이지만, **RGB 데이터는 여전히 존재**.
4DGS는 Alpha를 무시하고 RGB만 사용하므로 배경 색상이 학습됨.

---

## 3. 해결 방안

### 방안 A: Alpha 채널 활용 (선택)
4DGS 코드를 패치하여 Alpha 채널을 읽고 활용

### 방안 B: 배경 RGB를 흰색으로 설정
Lego 데이터셋처럼 배경 영역의 RGB를 흰색으로 설정

### 최종 선택: **A + B 조합**

---

## 4. 구현

### 4.1 배경 RGB를 흰색으로 설정 (`background_remover.py`)

```python
# 변경 전
rgba = np.dstack([rgb, mask])

# 변경 후
mask_binary = (mask > 127).astype(np.float32)[:, :, np.newaxis]
white_bg = np.ones_like(rgb) * 255
rgb_composited = (rgb * mask_binary + white_bg * (1 - mask_binary)).astype(np.uint8)
rgba = np.dstack([rgb_composited, mask])
```

**효과**: 배경 픽셀의 RGB가 [255, 255, 255]로 설정됨

### 4.2 Alpha 채널 추출 (`patch_4dgs_alpha.py` → `camera_utils.py`)

```python
# 패치 후
def loadCam(args, id, cam_info, resolution_scale):
    gt_alpha_mask = None
    if cam_info.image.shape[0] == 4:
        gt_alpha_mask = cam_info.image[3:4, :, :].clone()
    return Camera(..., gt_alpha_mask=gt_alpha_mask, ...)
```

### 4.3 GT 이미지 흰색 배경 합성 (`patch_4dgs_alpha.py` → `cameras.py`)

```python
# 패치 후
if gt_alpha_mask is not None:
    # RGB * alpha + white * (1 - alpha)
    self.original_image = self.original_image * gt_alpha_mask + (1.0 - gt_alpha_mask)
```

**효과**: GT 이미지가 `--white_background` 렌더링 결과와 일치

---

## 5. 결과

### 학습 명령어
```bash
python manage.py train data/scene_alpha --extra="--white_background"
```

### 비교

| 항목 | 패치 전 | 패치 후 |
|------|---------|---------|
| 배경 RGB | 원본 색상 유지 | 흰색 (255) |
| Alpha 활용 | ❌ 무시 | ✅ 활용 |
| GT-Render 매칭 | ❌ 불일치 | ✅ 일치 |
| 배경 Gaussian | ⚠️ 생성됨 | ✅ 감소 |

---

## 6. 한계 및 후속 실험

이 패치만으로는 배경 Gaussian이 완전히 제거되지 않음.
→ **실험 02: Loss Masking**으로 해결
