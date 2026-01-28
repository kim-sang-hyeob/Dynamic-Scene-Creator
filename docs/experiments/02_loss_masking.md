# 실험 02: Loss Masking으로 배경 Gaussian 제거

## 실험 정보
- **날짜**: 2025-01-28
- **관련 파일**: `patch_4dgs_alpha.py` (train.py 패치 부분)
- **선행 실험**: 실험 01 (Alpha Channel Handling)
- **상태**: ✅ 해결됨 (단, 실험 03과 함께 사용 필요)

---

## 1. 문제 발견

### 현상
실험 01의 Alpha 패치 적용 후에도 배경에 Gaussian이 형성됨.

### 원인
GT 이미지와 렌더링 결과가 둘 다 흰색 배경이더라도:
1. 초기 Gaussian이 배경 영역에 존재
2. 학습 과정에서 약간의 색상 차이로 인해 gradient 발생
3. 배경에도 Gaussian이 형성/유지됨

---

## 2. 해결 아이디어

**Loss 계산에서 배경 픽셀을 완전히 제외**

```
기존: Loss = |Render - GT| (전체 픽셀)
개선: Loss = |Render - GT| * mask (전경 픽셀만)
```

배경 픽셀의 Loss가 0이면:
- Gradient = 0
- 배경 Gaussian 업데이트 없음
- 결국 pruning으로 제거됨

---

## 3. 구현

### 3.1 Alpha Mask 수집 (`train.py`)

```python
# render loop 내부
alpha_masks = []
for viewpoint_cam in viewpoint_cams:
    ...
    if hasattr(viewpoint_cam, 'gt_alpha_mask') and viewpoint_cam.gt_alpha_mask is not None:
        alpha_masks.append(viewpoint_cam.gt_alpha_mask.cuda().unsqueeze(0))
    else:
        alpha_masks.append(None)
```

### 3.2 Masked L1 Loss

```python
# 기존
Ll1 = l1_loss(image_tensor, gt_image_tensor[:,:3,:,:])

# 패치 후
combined_mask = torch.cat([m for m in alpha_masks if m is not None], 0)
mask_binary = (combined_mask > 0.5).float()
masked_render = image_tensor * mask_binary
masked_gt = gt_image_tensor[:,:3,:,:] * mask_binary
valid_pixels = mask_binary.sum() + 1e-8
Ll1 = torch.abs(masked_render - masked_gt).sum() / valid_pixels
```

### 3.3 Masked SSIM Loss

```python
# 기존
ssim_loss = ssim(image_tensor, gt_image_tensor)

# 패치 후
if combined_mask is not None:
    mask_binary = (combined_mask > 0.5).float()
    ssim_loss = ssim(image_tensor * mask_binary, gt_image_tensor[:,:3,:,:] * mask_binary)
```

---

## 4. Loss Masking의 원리

### 4.1 Gradient 흐름

```
Forward:  Gaussian → Render → Loss
Backward: Gaussian ← Render ← Loss (gradient)

mask = 0인 픽셀:
  - Loss = 0
  - Gradient = 0
  - 해당 영역의 Gaussian 업데이트 없음
```

### 4.2 Densification에 미치는 영향

```python
# 4DGS densification 조건
if grad > threshold:
    split_or_clone()
```

배경 Gaussian은 gradient가 0이므로:
- Densification 대상에서 제외
- 기존 배경 Gaussian도 점점 pruning됨

---

## 5. 실험 결과

### 예상 vs 실제

| 항목 | 예상 | 실제 |
|------|------|------|
| 배경 Loss | 0 | 0 |
| 배경 Gradient | 0 | 0 |
| 배경 Gaussian | 감소 | ⚠️ 문제 발생 |

### 문제 발생

```
Training progress: 100%|███| 30000/30000 [16:53<00:00]
Loss=0.0176630, psnr=18.40, point=1080

[ITER 3000]  PSNR 18.388
[ITER 7000]  PSNR 18.388  ← 동일!
[ITER 14000] PSNR 18.388  ← 동일!
```

**PSNR이 전혀 변하지 않음** = 학습이 안 됨

---

## 6. 원인 분석

### 닭과 달걀 문제 (Chicken-and-Egg Problem)

```
초기 상태:
  - 초기 Gaussian: 랜덤 위치 (배경 포함)
  - 고양이 영역: Gaussian 없음

Loss Masking 적용:
  - 배경 픽셀: Loss = 0, Gradient = 0 ✓
  - 전경 픽셀: Loss ≠ 0, 하지만...
    - 전경에 Gaussian이 없으면 Render = 배경색
    - Gradient가 있어도 이동할 Gaussian이 없음

결과:
  - 아무것도 학습되지 않음
  - PSNR 고정
```

### 핵심 문제

```
Loss Masking은 "전경에 이미 Gaussian이 있을 때"만 작동
초기 Gaussian이 전경에 없으면 학습 불가능
```

---

## 7. 해결 방향

**→ 실험 03: Foreground-based Point Initialization**

초기 Gaussian을 전경(고양이) 영역에 배치하면:
1. 전경에 Gaussian 존재
2. Loss Masking이 정상 작동
3. Gradient로 Gaussian 업데이트
4. 학습 진행

---

## 8. 결론

| 조건 | Loss Masking 효과 |
|------|------------------|
| 초기 점이 전경에 있음 | ✅ 배경 제거 성공 |
| 초기 점이 랜덤 (배경 포함) | ❌ 학습 실패 |

**Loss Masking은 단독으로 사용 불가**, 반드시 **Foreground Initialization과 함께** 사용해야 함.
