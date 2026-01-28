# 실험 03: Foreground 기반 초기 Point Cloud 생성

## 실험 정보
- **날짜**: 2025-01-28
- **관련 파일**: `json_sync_utils.py` (`write_colmap_text` 함수)
- **선행 실험**: 실험 02 (Loss Masking)
- **상태**: ✅ 해결됨

---

## 1. 문제 정의

### 실험 02의 실패 원인
Loss Masking을 적용했으나 학습이 진행되지 않음.

**근본 원인**: 초기 Gaussian이 전경(고양이) 영역에 없음

```
초기 Point Cloud 분포:
  X range: -0.7 ~ 3.0
  Y range: -0.1 ~ 0.1  ← 거의 평면
  Z range: -8.1 ~ 7.4

문제: 이 점들이 고양이가 보이는 영역과 무관
```

### 기존 초기화 방식

```python
# patch_4dgs_sfm_free.py
xyz = np.random.randn(num_pts, 3) * 0.5  # 완전 랜덤
```

또는

```python
# json_sync_utils.py (수정 전)
for frame in frames:
    obj_pos = frame['objPos']  # 물체 중심 근처
    for dx, dy, dz in grid_3x3x3:
        points.append(obj_pos + offset)
```

두 방식 모두 **실제 고양이가 보이는 픽셀 위치와 무관**

---

## 2. 해결 아이디어

### Alpha Mask 활용

```
알고 있는 정보:
1. 각 프레임의 Alpha Mask (고양이 = 255, 배경 = 0)
2. 각 프레임의 카메라 Pose (Position, Rotation)
3. 카메라 Intrinsics (Focal Length, Principal Point)

아이디어:
  Alpha > 127인 픽셀 → 고양이가 있는 픽셀
  이 픽셀들을 3D로 Back-project → 고양이 근처의 3D 점
```

### Back-Projection 원리

```
2D 픽셀 (u, v) → 3D 점 (X, Y, Z)

1. 픽셀을 정규화 좌표로 변환
   x_norm = (u - cx) / focal
   y_norm = (v - cy) / focal

2. 카메라 공간에서 ray 방향 계산
   ray_cam = normalize([x_norm, y_norm, 1.0])

3. 월드 공간으로 변환
   ray_world = R_camera @ ray_cam

4. 추정 깊이에서 3D 점 생성
   point_3d = camera_position + ray_world * depth
```

---

## 3. 구현

### 3.1 주요 코드 (`json_sync_utils.py`)

```python
def write_colmap_text(frames, output_dir, img_dir, map_transform):
    ...
    if has_alpha:
        print("[COLMAP] Alpha channel detected - will use foreground-based point initialization")

        # ~5개 프레임 샘플링
        sample_frames = frames[::max(1, len(frames)//5)]

        for frame in sample_frames:
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            alpha = img[:, :, 3]
            foreground_mask = alpha > 127

            # 전경 픽셀 좌표 추출
            fy, fx = np.where(foreground_mask)

            # 프레임당 최대 200개 샘플링
            n_samples = min(200, len(fx))
            indices = np.random.choice(len(fx), n_samples, replace=False)

            # 카메라 pose 가져오기
            R_nerf, C_nerf = get_camera_pose(frame, map_transform)

            # 깊이 추정 (카메라-물체 거리)
            estimated_depth = np.linalg.norm(obj_nerf - C_nerf)

            # Back-projection
            for px, py in zip(sampled_x, sampled_y):
                x_norm = (px - cx) / focal
                y_norm = (py - cy) / focal
                ray_dir = normalize([x_norm, y_norm, 1.0])
                ray_world = R_nerf @ ray_dir

                # 깊이에 랜덤 변동 추가 (0.8 ~ 1.2배)
                depth = estimated_depth * (0.8 + 0.4 * random())
                point_3d = C_nerf + ray_world * depth

                all_points.append(point_3d)
```

### 3.2 설계 선택

| 항목 | 선택 | 이유 |
|------|------|------|
| 샘플 프레임 수 | ~5개 | 전체 프레임 사용 시 점이 너무 많음 |
| 프레임당 샘플 | 200개 | 적당한 밀도 + 계산 효율 |
| 깊이 변동 | ±20% | 단일 깊이면 평면에 점이 몰림 |
| 중복 제거 | 좌표 반올림 | 비슷한 위치의 점 제거 |

---

## 4. 실험 결과

### 초기 Point Cloud 비교

**수정 전 (Object Position 기반)**
```
Points: 1080
X range: -0.7 ~ 3.0
Y range: -0.1 ~ 0.1  ← 평면에 집중
Z range: -8.1 ~ 7.4
```

**수정 후 (Foreground 기반)**
```
Points: 1000
X range: -5.3 ~ 2.9
Y range: 3.1 ~ 5.1   ← 실제 고양이 높이
Z range: -2.1 ~ 13.9
```

### 학습 진행 비교

**수정 전**
```
Iter 3000:  PSNR 18.38, Points 1080
Iter 7000:  PSNR 18.38, Points 1080  ← 변화 없음
Iter 14000: PSNR 18.38, Points 1080  ← 변화 없음
```

**수정 후**
```
Iter 3000:  PSNR 20.46, Points 9427   ← Densify 시작
Iter 3000:  PSNR 30.60, Points 15885  ← 급격한 개선
Iter 6000:  PSNR 33.31, Points 19041  ← 계속 개선
```

### 핵심 지표 변화

| 지표 | 수정 전 | 수정 후 | 변화 |
|------|---------|---------|------|
| 최종 PSNR | 18.4 | 33+ | +15 |
| Point 증가 | 없음 | 1000 → 19000+ | 19배 |
| 학습 여부 | ❌ | ✅ | - |

---

## 5. 시각화

### Back-Projection 과정

```
Frame 0 (t=0.0)           Frame 20 (t=0.5)
┌─────────────────┐       ┌─────────────────┐
│    ┌───┐        │       │        ┌───┐    │
│    │ 🐱│        │       │        │ 🐱│    │
│    └───┘        │       │        └───┘    │
│  Camera →       │       │      ← Camera   │
└─────────────────┘       └─────────────────┘
        ↓                         ↓
   Back-project              Back-project
        ↓                         ↓
        ┌─────────────────────────┐
        │    3D Point Cloud       │
        │         🐱              │
        │    (다양한 깊이)          │
        └─────────────────────────┘
```

---

## 6. 결론

### 해결된 문제

1. **닭과 달걀 문제 해결**
   - 초기 점이 전경에 존재 → Loss Masking 정상 작동

2. **Densification 활성화**
   - 전경 점에서 gradient 발생 → 점 분할/복제

3. **학습 성공**
   - PSNR 18 → 33 (거의 2배)

### 필수 조합

```
배경 없는 4DGS 학습 =
  Alpha Channel Handling (실험 01)
  + Loss Masking (실험 02)
  + Foreground Point Initialization (실험 03)
```

세 가지가 모두 적용되어야 배경 없는 깨끗한 결과 획득 가능.
