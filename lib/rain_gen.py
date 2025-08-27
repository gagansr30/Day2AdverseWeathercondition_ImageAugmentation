import os
import argparse
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import random
import gc

def add_fog_rain_wet_effect_from_images(original_img, depth_img,
                            fog_color=(220, 220, 220), rain_color=(255, 255, 255),
                            fog_intensity=1.0, rain_intensity=1.0,
                            rain_density=1000, rain_opacity=100,
                            wet_darkening=0.4):
    original = original_img.convert("RGB")
    depth = depth_img.convert("L").resize(original.size)
    width, height = original.size

    orig_np = np.array(original).astype(np.float32)
    depth_np = np.array(depth).astype(np.float32) / 255.0
    depth_np = depth_np ** 1.5
    depth_mask = depth_np[..., None]

    fog_color_np = np.ones_like(orig_np) * np.array(fog_color, dtype=np.float32)
    fogged_np = orig_np * (1 - depth_mask * fog_intensity) + fog_color_np * (depth_mask * fog_intensity)

    wet_mask = np.linspace(1.0, 1.0 - wet_darkening, height).reshape(-1, 1)
    wet_mask = np.repeat(wet_mask, width, axis=1)[..., None]
    fogged_wet_np = fogged_np * wet_mask
    fogged_wet_np = np.clip(fogged_wet_np, 0, 255).astype(np.uint8)
    fogged_img = Image.fromarray(fogged_wet_np)

    rain_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rain_layer)

    for _ in range(int(rain_density * rain_intensity)):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        pixel_depth = depth_np[y, x]
        if pixel_depth < 0.05:
            continue
        length = random.randint(6, 12)
        angle = random.choice([-1, 0, 1])
        end_x = x + angle
        end_y = y + length
        opacity = int(rain_opacity * pixel_depth)
        if opacity > 5:
            draw.line([(x, y), (end_x, end_y)], fill=(*rain_color, opacity), width=1)

    rain_layer = rain_layer.filter(ImageFilter.GaussianBlur(radius=0.3))
    final = Image.alpha_composite(fogged_img.convert("RGBA"), rain_layer).convert("RGB")
    return final

def main():
    parser = argparse.ArgumentParser(description="Fog + Rain + Wet Effect Generator")
    parser.add_argument("--rgb_dir", required=True, help="Directory with RGB images")
    parser.add_argument("--depth_dir", required=True, help="Directory with depth images")
    parser.add_argument("--output_dir", required=True, help="Directory to save outputs")
    parser.add_argument("--fog_intensity", type=float, default=0.3)
    parser.add_argument("--rain_intensity", type=float, default=0.8)
    parser.add_argument("--rain_density", type=int, default=3000)
    parser.add_argument("--rain_opacity", type=int, default=150)
    parser.add_argument("--wet_darkening", type=float, default=0.4)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    for file in os.listdir(args.rgb_dir):
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            base = os.path.splitext(file)[0]
            rgb_path = os.path.join(args.rgb_dir, file)
            depth_path = os.path.join(args.depth_dir, f"{base}_depth.jpg")

            if not os.path.exists(depth_path):
                print(f" Skipping {file}: depth image not found")
                continue

            print(f" Processing {file} with {base}_depth.jpg")

            rgb_img = Image.open(rgb_path)
            depth_img = Image.open(depth_path)

            result = add_fog_rain_wet_effect_from_images(
                rgb_img, depth_img,
                fog_intensity=args.fog_intensity,
                rain_intensity=args.rain_intensity,
                rain_density=args.rain_density,
                rain_opacity=args.rain_opacity,
                wet_darkening=args.wet_darkening
            )

            output_path = os.path.join(args.output_dir, f"{base}_prerain.jpg")
            result.save(output_path, format="JPEG")
            print(f" Saved: {output_path}")

            gc.collect()

    print("\n All fog/rain/wet enhancements complete.")

if __name__ == "__main__":
    main()
