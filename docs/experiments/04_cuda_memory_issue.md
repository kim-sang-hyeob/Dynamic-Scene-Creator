# 실험 04: CUDA Memory Access 오류 해결

## 실험 정보
- **날짜**: 2025-01-28
- **관련 파일**: `patch_4dgs_alpha.py` (`camera_utils.py` 패치)
- **상태**: ✅ 해결됨

---

## 1. 문제 발견

### 현상
학습 중 무작위 iteration에서 CUDA 오류 발생 후 crash.

```
Training progress: 40%|████████| 12110/30000
CUDA error: an illegal memory access was encountered
```

### 특징
- 발생 시점 불규칙 (5000, 12000, 8000 등)
- `--densify_until_iter` 조정해도 발생
- Point 수가 적어도 (3000개) 발생

---

## 2. 원인 분석

### 가설 1: GPU 메모리 부족
**검증**: Point 수 제한 (`--densify_until_iter 5000`)

```
Points: 3072 (매우 적음)
결과: 여전히 crash
```

**결과**: ❌ 메모리 부족이 아님

### 가설 2: Tensor View 문제
**검증**: Alpha mask 추출 코드 분석

```python
# 문제 코드
gt_alpha_mask = cam_info.image[3:4, :, :]  # View (참조)
```

**문제점**:
```
cam_info.image (원본 텐서)
    ↓
gt_alpha_mask (View, 원본의 일부를 참조)
    ↓
원본 텐서 garbage collected
    ↓
gt_alpha_mask가 참조하는 메모리 해제됨
    ↓
나중에 접근 시 illegal memory access
```

**결과**: ✅ **Tensor View가 원인**

---

## 3. PyTorch View vs Clone

### View (참조)
```python
a = torch.tensor([1, 2, 3, 4])
b = a[1:3]  # View: 같은 메모리 공유

a.data_ptr() == b.data_ptr()  # 내부적으로 같은 메모리
```

### Clone (복사)
```python
a = torch.tensor([1, 2, 3, 4])
b = a[1:3].clone()  # Clone: 독립적인 메모리

a.data_ptr() != b.data_ptr()  # 다른 메모리
```

### 문제 상황

```
1. cam_info 로드
2. gt_alpha_mask = cam_info.image[3:4, :, :] (View)
3. cam_info 사용 완료 → garbage collection 대기
4. ... 다른 연산 ...
5. Python GC가 cam_info.image 해제
6. gt_alpha_mask 접근 시 → 해제된 메모리 접근 → CRASH
```

---

## 4. 해결

### 수정 코드

```python
# 수정 전 (View)
gt_alpha_mask = cam_info.image[3:4, :, :]

# 수정 후 (Clone)
gt_alpha_mask = cam_info.image[3:4, :, :].clone()
```

### `.clone()` 효과

```
cam_info.image ──┐
                 ├── [3:4, :, :] → 새로운 메모리에 복사
                 │                       ↓
                 │               gt_alpha_mask (독립)
                 │
cam_info 해제 ───┘
                        gt_alpha_mask는 영향 없음 ✓
```

---

## 5. 교훈

### PyTorch 메모리 관리 주의사항

1. **슬라이싱은 View를 생성**
   ```python
   x = tensor[a:b]  # View
   x = tensor[a:b].clone()  # Clone (안전)
   ```

2. **장기 보관 시 Clone 사용**
   ```python
   # 나쁜 예
   self.cached = some_tensor[idx]  # View, 원본 해제되면 위험

   # 좋은 예
   self.cached = some_tensor[idx].clone()  # Clone, 안전
   ```

3. **디버깅 팁**
   - `illegal memory access` → View/Clone 확인
   - 무작위 crash → GC 타이밍 문제 의심

---

## 6. 관련 코드 위치

```python
# external/4dgs/utils/camera_utils.py
def loadCam(args, id, cam_info, resolution_scale):
    gt_alpha_mask = None
    if cam_info.image.shape[0] == 4:
        gt_alpha_mask = cam_info.image[3:4, :, :].clone()  # ← .clone() 추가
    ...
```

---

## 7. 결론

| 수정 전 | 수정 후 |
|---------|---------|
| View (메모리 공유) | Clone (독립 메모리) |
| 무작위 crash | 안정적 학습 |
| Illegal memory access | 오류 없음 |

**교훈**: 텐서를 장기 보관하거나 원본 해제 후에도 사용할 경우 `.clone()` 필수.
