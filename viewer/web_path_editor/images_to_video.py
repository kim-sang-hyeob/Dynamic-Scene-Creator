#!/usr/bin/env python3
"""
이미지 시퀀스를 동영상으로 변환하는 스크립트

사용법:
    python images_to_video.py                          # 기본 설정으로 실행
    python images_to_video.py -i ./images -o out.mp4   # 입력/출력 지정
    python images_to_video.py --fps 30                 # FPS 지정
    python images_to_video.py --pattern "frame_*.png"  # 파일 패턴 지정
"""

import argparse
import glob
import os
import re
import subprocess
import sys


def natural_sort_key(s):
    """자연 정렬을 위한 키 함수 (frame_1, frame_2, ..., frame_10 순서)"""
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def create_video_ffmpeg(image_dir, output_path, fps, pattern):
    """FFmpeg를 사용하여 동영상 생성"""
    
    # 이미지 파일 찾기
    image_pattern = os.path.join(image_dir, pattern)
    images = sorted(glob.glob(image_pattern), key=natural_sort_key)
    
    if not images:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_pattern}")
        return False
    
    print(f"=" * 50)
    print(f"📸 이미지 → 동영상 변환")
    print(f"=" * 50)
    print(f"입력 폴더: {image_dir}")
    print(f"이미지 수: {len(images)}개")
    print(f"FPS: {fps}")
    print(f"예상 길이: {len(images) / fps:.2f}초")
    print(f"출력 파일: {output_path}")
    print(f"=" * 50)
    
    # FFmpeg 명령어 구성
    # -framerate: 입력 프레임레이트
    # -pattern_type glob: 글로브 패턴 사용
    # -i: 입력 패턴
    # -c:v libx264: H.264 코덱
    # -pix_fmt yuv420p: 호환성 있는 픽셀 포맷
    # -crf 18: 품질 (0=무손실, 23=기본, 51=최저)
    
    # FFmpeg는 패턴으로 입력받기
    # frame_%04d.png 형식 사용
    
    # 첫 번째 이미지에서 패턴 추출
    first_image = os.path.basename(images[0])
    # frame_0000.png -> frame_%04d.png
    ffmpeg_pattern = re.sub(r'\d+', lambda m: f'%0{len(m.group())}d', first_image, count=1)
    input_pattern = os.path.join(image_dir, ffmpeg_pattern)
    
    cmd = [
        'ffmpeg',
        '-y',  # 덮어쓰기
        '-framerate', str(fps),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '18',
        output_path
    ]
    
    print(f"실행 명령어: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ 동영상 생성 완료: {output_path}")
            # 파일 크기 확인
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"  파일 크기: {size_mb:.2f} MB")
            return True
        else:
            print(f"❌ FFmpeg 오류:")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("❌ FFmpeg가 설치되어 있지 않습니다.")
        print("   설치: sudo apt install ffmpeg")
        return False


def create_video_opencv(image_dir, output_path, fps, pattern):
    """OpenCV를 사용하여 동영상 생성 (FFmpeg 없을 때 대안)"""
    try:
        import cv2
    except ImportError:
        print("❌ OpenCV가 설치되어 있지 않습니다.")
        print("   설치: pip install opencv-python")
        return False
    
    # 이미지 파일 찾기
    image_pattern = os.path.join(image_dir, pattern)
    images = sorted(glob.glob(image_pattern), key=natural_sort_key)
    
    if not images:
        print(f"❌ 이미지를 찾을 수 없습니다: {image_pattern}")
        return False
    
    print(f"=" * 50)
    print(f"📸 이미지 → 동영상 변환 (OpenCV)")
    print(f"=" * 50)
    print(f"입력 폴더: {image_dir}")
    print(f"이미지 수: {len(images)}개")
    print(f"FPS: {fps}")
    print(f"예상 길이: {len(images) / fps:.2f}초")
    print(f"출력 파일: {output_path}")
    print(f"=" * 50)
    
    # 첫 이미지로 크기 확인
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"❌ 이미지를 읽을 수 없습니다: {images[0]}")
        return False
    
    height, width, _ = first_frame.shape
    print(f"해상도: {width}x{height}")
    
    # VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for i, img_path in enumerate(images):
        frame = cv2.imread(img_path)
        if frame is not None:
            out.write(frame)
        if (i + 1) % 50 == 0:
            print(f"  처리 중: {i + 1}/{len(images)}")
    
    out.release()
    print(f"✓ 동영상 생성 완료: {output_path}")
    
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  파일 크기: {size_mb:.2f} MB")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='이미지 시퀀스를 동영상으로 변환',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python images_to_video.py -i ./images -o output.mp4 --fps 30
  python images_to_video.py --pattern "*.jpg" --fps 24
        """
    )
    
    parser.add_argument('-i', '--input', default='./images',
                        help='이미지 폴더 경로 (기본: ./images)')
    parser.add_argument('-o', '--output', default='./output.mp4',
                        help='출력 동영상 경로 (기본: ./output.mp4)')
    parser.add_argument('--fps', type=float, default=21,
                        help='프레임 레이트 (기본: 21)')
    parser.add_argument('--pattern', default='frame_*.png',
                        help='이미지 파일 패턴 (기본: frame_*.png)')
    parser.add_argument('--use-opencv', action='store_true',
                        help='FFmpeg 대신 OpenCV 사용')
    
    args = parser.parse_args()
    
    # 절대 경로로 변환
    input_dir = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    if not os.path.isdir(input_dir):
        print(f"❌ 폴더가 존재하지 않습니다: {input_dir}")
        sys.exit(1)
    
    # 동영상 생성
    if args.use_opencv:
        success = create_video_opencv(input_dir, output_path, args.fps, args.pattern)
    else:
        success = create_video_ffmpeg(input_dir, output_path, args.fps, args.pattern)
        if not success:
            print("\nFFmpeg 실패, OpenCV로 재시도...")
            success = create_video_opencv(input_dir, output_path, args.fps, args.pattern)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
