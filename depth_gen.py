# ===== IMPORTS =====
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation
from scipy.ndimage import gaussian_filter
import argparse
import warnings
import os

warnings.filterwarnings("ignore")

# ===== DEVICE SETUP =====
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# ===== PROFESSIONAL DEPTH CONVERTER =====
class ProfessionalDepthConverter:
    def __init__(self, model_name="dpt-large"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        print(f"Loading {self.model_name} model...")
        try:
            if self.model_name == "dpt-large":
                self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
                self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            else:
                self.model = pipeline("depth-estimation", model="Intel/MiDaS", device=0 if self.device == "cuda" else -1)
            print(" Model loaded successfully!")
        except Exception as e:
            print(f" Loading failed, fallback to MiDaS: {e}")
            self.model = pipeline("depth-estimation", model="Intel/MiDaS", device=0 if self.device == "cuda" else -1)
            self.model_name = "midas"

    def load_image(self, image_path, max_size=512):
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            image = image.resize((int(width * ratio), int(height * ratio)), Image.Resampling.LANCZOS)
        return image

    def estimate_raw_depth(self, image):
        try:
            if self.model_name == "midas":
                result = self.model(image)
                return np.array(result["depth"])
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth
                prediction = F.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                return prediction
        except Exception as e:
            print(f" Depth estimation failed: {e}")
            return None

    def create_professional_depth(self, image, smoothing=2.0, contrast_boost=1.5, gradient_power=1.2, edge_preservation=0.8):
        print(" Creating professional depth map...")
        if isinstance(image, str):
            image = self.load_image(image)

        raw_depth = self.estimate_raw_depth(image)
        if raw_depth is None:
            return None

        raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        depth = self.invert_depth(raw_depth)
        depth = self.apply_smoothing(depth, smoothing, edge_preservation)
        depth = self.enhance_contrast(depth, contrast_boost)
        depth = self.apply_gradient_curve(depth, gradient_power)
        depth = self.final_normalization(depth)
        print(" Professional depth map created!")
        return depth

    def invert_depth(self, depth):
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)
        return 1.0 - normalized

    def apply_smoothing(self, depth, strength, edge_preservation):
        if strength <= 0:
            return depth
        depth_uint8 = (depth * 255).astype(np.uint8)
        if edge_preservation > 0:
            d = int(strength * 4)
            sigma_color = edge_preservation * 50
            sigma_space = strength * 20
            smoothed = cv2.bilateralFilter(depth_uint8, d, sigma_color, sigma_space)
        else:
            smoothed = gaussian_filter(depth_uint8, sigma=strength)
        return smoothed.astype(np.float32) / 255.0

    def enhance_contrast(self, depth, contrast_boost):
        if contrast_boost <= 1.0:
            return depth
        p1, p99 = np.percentile(depth, (1, 99))
        clipped = np.clip(depth, p1, p99)
        normalized = (clipped - p1) / (p99 - p1) if p99 > p1 else clipped
        return np.power(normalized, 1.0 / contrast_boost)

    def apply_gradient_curve(self, depth, power):
        return np.power(depth, power) if power != 1.0 else depth

    def final_normalization(self, depth):
        d_min, d_max = depth.min(), depth.max()
        normalized = (depth - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth)
        return (normalized * 255).astype(np.uint8)

# ===== DISPLAY & SAVE FUNCTIONS =====
def display_and_save_result(original_image, depth_map, save_path="professional_depth.png"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(depth_map, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Professional Depth Map", fontsize=14, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # Save result
    Image.fromarray(depth_map).save(save_path, optimize=True, quality=95)
    print(f" Saved depth map as: {save_path}")

# ===== MAIN EXECUTION =====
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, help="Single image path")
    parser.add_argument('--input_dir', type=str, help="Folder containing multiple images")
    parser.add_argument('--output_dir', type=str, default="depth_imgs", help="Folder to save depth maps")
    args = parser.parse_args()

    converter = ProfessionalDepthConverter()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.input_file:
        # Process single file
        original = converter.load_image(args.input_file)
        depth_map = converter.create_professional_depth(original)
        output_path = os.path.join(args.output_dir, "depth_map.png")
        Image.fromarray(depth_map).save(output_path)
        print(f" Saved: {output_path}")

    elif args.input_dir:
        # Batch process all .jpg, .jpeg, .png files
        supported_exts = (".jpg", ".jpeg", ".png")
        image_paths = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                       if f.lower().endswith(supported_exts)]

        print(f" Found {len(image_paths)} images in {args.input_dir}")
        for i, path in enumerate(image_paths, 1):
            try:
                print(f"\n [{i}/{len(image_paths)}] Processing: {os.path.basename(path)}")
                image = converter.load_image(path)
                depth_map = converter.create_professional_depth(image)

                base = os.path.splitext(os.path.basename(path))[0]
                output_path = os.path.join(args.output_dir, f"{base}_depth.jpg")
                Image.fromarray(depth_map).save(output_path, quality=95)

                print(f" Saved: {output_path}")
            except Exception as e:
                print(f" Error processing {path}: {e}")
    else:
        print(" Please specify either --input_file or --input_dir")
