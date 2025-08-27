# fog_gen.py

import os
import argparse
from PIL import Image
import numpy as np

def add_fog_from_depth_images(original_img, depth_img, fog_color=(220, 220, 220), fog_intensity=0.4):
    fog_intensity = max(0.0, min(0.6, fog_intensity))

    original = original_img.convert("RGB")
    depth = depth_img.convert("L").resize(original.size)

    orig_np = np.array(original).astype(np.float32)
    depth_np = np.array(depth).astype(np.float32) / 255.0  # white = far = more fog
    depth_mask = depth_np[..., None] * fog_intensity

    fog_color_np = np.ones_like(orig_np) * np.array(fog_color, dtype=np.float32)
    fogged = orig_np * (1 - depth_mask) + fog_color_np * depth_mask
    fogged = np.clip(fogged, 0, 255).astype(np.uint8)

    return Image.fromarray(fogged)

def main(rgb_dir, depth_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    rgb_files = sorted([f for f in os.listdir(rgb_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    print(f" Found {len(rgb_files)} RGB images in {rgb_dir}")

    for i, rgb_file in enumerate(rgb_files, 1):
        base_name = os.path.splitext(rgb_file)[0]
        depth_file = f"{base_name}_depth.jpg"
        rgb_path = os.path.join(rgb_dir, rgb_file)
        depth_path = os.path.join(depth_dir, depth_file)

        if not os.path.exists(depth_path):
            print(f" Skipping {rgb_file} — no matching depth file found.")
            continue

        try:
            rgb_image = Image.open(rgb_path)
            depth_image = Image.open(depth_path)

            foggy = add_fog_from_depth_images(rgb_image, depth_image, fog_intensity=1.0)

            output_path = os.path.join(output_dir, f"{base_name}_prefog.jpg")
            foggy.save(output_path, quality=95)
            print(f" Saved: {output_path}")
        except Exception as e:
            print(f" Error processing {rgb_file}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb_dir", required=True, help="Path to folder with RGB images")
    parser.add_argument("--depth_dir", required=True, help="Path to folder with corresponding depth maps")
    parser.add_argument("--output_dir", required=True, help="Where to save foggy images")

    args = parser.parse_args()
    main(args.rgb_dir, args.depth_dir, args.output_dir)
