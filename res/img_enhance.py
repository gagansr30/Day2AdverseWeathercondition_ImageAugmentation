import os
import argparse
from PIL import Image
from datetime import datetime
import torch
import gc
from diffusers import StableDiffusionInstructPix2PixPipeline
import warnings

warnings.filterwarnings("ignore")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.empty_cache()

class RainEffectEnhancer:
    def __init__(self):
        self.device = device
        self.pipe = None
        self.model_loaded = False

    def load_model(self):
        if self.model_loaded:
            return True

        try:
            self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                "timbrooks/instruct-pix2pix",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                requires_safety_checker=False
            ).to(self.device)

            if self.device == "cuda":
                try:
                    self.pipe.enable_model_cpu_offload()
                except:
                    pass
                try:
                    self.pipe.enable_attention_slicing()
                except:
                    pass

            self.model_loaded = True
            self.clear_memory()
            return True

        except Exception as e:
            print(f" Model loading failed: {e}")
            return False

    def clear_memory(self):
        if self.device == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    def load_image(self, image_path, max_size=512):
        try:
            image = Image.open(image_path).convert("RGB")
            width, height = image.size
            if max(width, height) > max_size:
                ratio = max_size / max(width, height)
                new_width = int(width * ratio) - (int(width * ratio) % 8)
                new_height = int(height * ratio) - (int(height * ratio) % 8)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            print(f" Image loading failed: {e}")
            return None

    def save_image(self, image, output_path):
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            image.save(output_path, optimize=True, quality=95)
            print(f" Saved: {output_path}")
        except Exception as e:
            print(f" Saving failed: {e}")

    def add_rain_effect(self, image, prompt="make it look like real fog", intensity=0.7, steps=25):
        if not self.model_loaded and not self.load_model():
            return None
        try:
            self.clear_memory()

            image_guidance = max(1.0, 1.8 - (intensity * 0.8))
            text_guidance = min(15.0, 7.0 + (intensity * 5.0))

            with torch.inference_mode():
                result = self.pipe(
                    prompt,
                    image=image,
                    num_inference_steps=steps,
                    image_guidance_scale=image_guidance,
                    guidance_scale=text_guidance,
                    generator=torch.manual_seed(42)
                ).images[0]

            self.clear_memory()
            return result

        except Exception as e:
            print(f" Error applying effect: {e}")
            return None

# ======================= MAIN SCRIPT =========================
def process_images(input_dir, output_dir, prompt="make it look like realistic", intensity=0.4):
    enhancer = RainEffectEnhancer()
    if not enhancer.load_model():
        return

    os.makedirs(output_dir, exist_ok=True)
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for fname in sorted(image_files):
        input_path = os.path.join(input_dir, fname)
        print(f" Processing: {input_path}")
        image = enhancer.load_image(input_path)
        if image is None:
            continue

        result = enhancer.add_rain_effect(image, prompt, intensity)
        if result is None:
            continue

        # Construct output filename: "1_prefog.jpg" -> "1_fog.jpg"
        out_fname = fname.replace("_pre", "_")


        output_path = os.path.join(output_dir, out_fname)
        enhancer.save_image(result, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fog Enhancer")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input prefog images")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save fog images")
    parser.add_argument("--prompt", type=str, default="make it look like real fog", help="Fog description")
    parser.add_argument("--intensity", type=float, default=0.4, help="Fog intensity (0.1 to 1.0)")
    args = parser.parse_args()

    process_images(args.input_dir, args.output_dir, args.prompt, args.intensity)
