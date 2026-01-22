from PIL import Image
import os

def split_image(input_path, output_dir):
    img = Image.open(input_path)
    width, height = img.size
    
    # Define quadrants
    w_mid = width // 2
    h_mid = height // 2
    
    quadrants = [
        (0, 0, w_mid, h_mid),        # Top-Left (0.png)
        (w_mid, 0, width, h_mid),    # Top-Right (1.png)
        (0, h_mid, w_mid, height),   # Bottom-Left (2.png)
        (w_mid, h_mid, width, height) # Bottom-Right (3.png)
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, box in enumerate(quadrants):
        quad = img.crop(box)
        # Optional: Crop further if there is too much white padding
        # quad = quad.crop(quad.getbbox()) 
        output_path = os.path.join(output_dir, f"{i}.png")
        quad.save(output_path)
        print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_img = "/Users/blakelee/Desktop/Semantic Segmentation Project/3dgs_project/data/seoul_landmarks/sample_seoul.jpeg"
    target_dir = "/Users/blakelee/Desktop/Semantic Segmentation Project/3dgs_project/data/seoul_landmarks/namsan/input"
    split_image(input_img, target_dir)
