# 4D Gaussians를 위한 export_to_splatv

이 도구는 학습된 4D Gaussian Splatting 모델을 웹에서 가볍게 시각화할 수 있는 `.splatv` 형식으로 변환해줍니다.

## 필수 조건

이 도구는 [4DGaussians](https://github.com/hustvl/4DGaussians) 환경 내에서 동작하도록 설계되었습니다.

## 설치 방법

1. `export_to_splatv.py` 파일을 4DGaussians 저장소의 최상위(root) 디렉토리로 복사하세요.
   ```bash
   cp export_to_splatv.py /path/to/4DGaussians/
   ```

2. (선택 사항) `web_viewer` 폴더는 원하는 위치 아무 곳에나 두어도 상관없습니다.

## 사용법

`4DGaussians` 저장소의 최상위 경로에서 스크립트를 실행하여 학습된 모델을 변환합니다.

명령어 구조:
```bash
python export_to_splatv.py -s <dataset_path> -m <model_output_path> --output <output_filename.splatv>
```

**인자 (Arguments):**
- `-s, --source_path`: 데이터셋 경로 (학습에 사용한 경로와 동일해야 함).
- `-m, --model_path`: 학습된 모델의 출력 디렉토리 경로 (이 경로 아래에 `point_cloud/iteration_XXXX/` 폴더가 있어야 하며, 그 안에 `point_cloud.ply`와 `deformation.pth` 등이 포함되어 있어야 합니다).
- `--output`: 저장할 `.splatv` 파일 경로.
- `--iteration`: (선택) 특정 반복(iteration)을 불러올 때 사용. 기본값은 가장 최신 모델을 불러옵니다.
- `--num_samples`: (선택) 모션을 피팅(fitting)할 시간 샘플 수. 기본값은 20입니다.

**예시:**
```bash
python export_to_splatv.py -s data/lego -m output/lego_4d --output full_lego.splatv
```

## 웹 뷰어 (Web Viewer)

`web_viewer` 디렉토리에 가벼운 웹 뷰어가 포함되어 있습니다.

1. **로컬 웹 서버 시작:**
   브라우저 보안 제한으로 인해 로컬 파일을 직접 불러오지 못할 수 있습니다. 간단한 HTTP 서버를 실행하는 것을 권장합니다.
   
   ```bash
   cd web_viewer
   python3 -m http.server 8000
   ```

2. **뷰어 열기:**
   웹 브라우저를 열고 다음 주소로 접속하세요: [http://localhost:8000](http://localhost:8000)

3. **모델 불러오기:**
   - 생성된 `.splatv` 파일을 브라우저 창으로 드래그 앤 드롭하세요.
   - **조작법:**
     - **왼쪽 클릭 + 드래그**: 궤도 회전 (Orbit/Rotate).
     - **오른쪽 클릭 + 드래그**: 화면 이동 (Pan).
     - **스크롤**: 줌 인/아웃 (Zoom).
     - **재생/일시정지**: UI 버튼을 사용하여 애니메이션을 제어합니다.
     - **속도**: 재생 속도를 조절합니다.
