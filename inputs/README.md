# inputs/

원본 입력 파일을 이 폴더에 저장하세요.

## 폴더 구조 예시

```
inputs/
└── black_cat/
    ├── output_cat.mp4      # Diffusion 생성 비디오
    ├── full_data.json      # Unity 카메라 트래킹 JSON
    └── original.mp4        # 원본 Unity 비디오
```

## 사용법

```bash
python manage.py process-unity \
    inputs/black_cat/output_cat.mp4 \
    inputs/black_cat/full_data.json \
    inputs/black_cat/original.mp4 \
    --output black_cat --frames 40 --resize 0.5
```

> 이 폴더의 내용물은 `.gitignore`에 의해 무시됩니다.
