

# ===== IMPORTS CELL =====
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from transformers import pipeline, DPTImageProcessor, DPTForDepthEstimation
from scipy import ndimage
from scipy.ndimage import gaussian_filter
import requests
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f" Using device: {device}")

# ===== PROFESSIONAL DEPTH CONVERTER CELL =====
class ProfessionalDepthConverter:
    """
    Creates professional depth maps with smooth gradients
    Exactly like your reference image
    """
    
    def __init__(self, model_name="dpt-large"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self.load_model()
    
    def load_model(self):
        """Load depth estimation model"""
        print(f"Loading {self.model_name} model...")
        
        try:
            if self.model_name == "dpt-large":
                self.processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
                self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
            else:
                # Fallback to MiDaS
                self.model = pipeline("depth-estimation", model="Intel/MiDaS", device=0 if self.device == "cuda" else -1)
            
            print(f" Model loaded successfully!")
        except Exception as e:
            print(f" Loading failed, using MiDaS fallback: {e}")
            self.model = pipeline("depth-estimation", model="Intel/MiDaS", device=0 if self.device == "cuda" else -1)
            self.model_name = "midas"
    
    def load_image(self, image_path, max_size=512):
        """Load and preprocess image"""
        if isinstance(image_path, str):
            if image_path.startswith(('http://', 'https://')):
                response = requests.get(image_path)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
        else:
            image = image_path
        
        # Resize while maintaining aspect ratio
        width, height = image.size
        if max(width, height) > max_size:
            ratio = max_size / max(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return image
    
    def estimate_raw_depth(self, image):
        """Get raw depth estimation"""
        try:
            if self.model_name == "midas":
                result = self.model(image)
                return np.array(result["depth"])
            else:
                inputs = self.processor(images=image, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predicted_depth = outputs.predicted_depth
                
                prediction = torch.nn.functional.interpolate(
                    predicted_depth.unsqueeze(1),
                    size=image.size[::-1],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze().cpu().numpy()
                
                return prediction
        except Exception as e:
            print(f" Depth estimation failed: {e}")
            return None
    
    def create_professional_depth(self, image, 
                                smoothing=2.0, 
                                contrast_boost=1.5, 
                                gradient_power=1.2,
                                edge_preservation=0.8):
        """
        Create professional depth map like your reference image
        
        Args:
            image: Input image
            smoothing: Gaussian smoothing strength (0.5-5.0)
            contrast_boost: Contrast enhancement (1.0-3.0)
            gradient_power: Power curve for gradients (0.5-2.0)
            edge_preservation: Edge preservation strength (0.0-1.0)
        """
        print(" Creating professional depth map...")
        
        # Load image
        if isinstance(image, str):
            image = self.load_image(image)
        
        # Get raw depth
        raw_depth = self.estimate_raw_depth(image)
        if raw_depth is None:
            return None
        
        # Clean up invalid values
        raw_depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Step 1: Invert for near=light, far=dark
        depth_inverted = self.invert_depth(raw_depth)
        
        # Step 2: Apply smoothing for gradients
        depth_smooth = self.apply_professional_smoothing(depth_inverted, smoothing, edge_preservation)
        
        # Step 3: Enhance contrast professionally
        depth_enhanced = self.enhance_contrast_professional(depth_smooth, contrast_boost)
        
        # Step 4: Apply gradient power curve
        depth_curved = self.apply_gradient_curve(depth_enhanced, gradient_power)
        
        # Step 5: Final normalization to perfect 0-255 range
        depth_final = self.final_normalization(depth_curved)
        
        print(" Professional depth map created!")
        return depth_final
    
    def invert_depth(self, depth):
        """Invert depth so near=high values, far=low values"""
        # Normalize to 0-1 first
        depth_min, depth_max = depth.min(), depth.max()
        if depth_max > depth_min:
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)
        
        # Invert: near objects become bright
        inverted = 1.0 - normalized
        return inverted
    
    def apply_professional_smoothing(self, depth, smoothing_strength, edge_preservation):
        """Apply smoothing while preserving important edges"""
        if smoothing_strength <= 0:
            return depth
        
        # Convert to uint8 for OpenCV processing
        depth_uint8 = (depth * 255).astype(np.uint8)
        
        # Apply bilateral filter for edge-preserving smoothing
        if edge_preservation > 0:
            # Bilateral filter preserves edges while smoothing
            d = int(smoothing_strength * 4)
            sigma_color = edge_preservation * 50
            sigma_space = smoothing_strength * 20
            
            smoothed = cv2.bilateralFilter(depth_uint8, d, sigma_color, sigma_space)
        else:
            # Pure Gaussian smoothing
            sigma = smoothing_strength
            smoothed = gaussian_filter(depth_uint8, sigma=sigma)
            smoothed = np.clip(smoothed, 0, 255).astype(np.uint8)
        
        # Convert back to float
        return smoothed.astype(np.float32) / 255.0
    
    def enhance_contrast_professional(self, depth, contrast_boost):
        """Professional contrast enhancement"""
        if contrast_boost <= 1.0:
            return depth
        
        # Use histogram stretching for better contrast
        # Get percentiles for robust stretching
        p1, p99 = np.percentile(depth, (1, 99))
        
        # Clip outliers
        depth_clipped = np.clip(depth, p1, p99)
        
        # Normalize to 0-1
        if p99 > p1:
            depth_normalized = (depth_clipped - p1) / (p99 - p1)
        else:
            depth_normalized = depth_clipped
        
        # Apply contrast boost using power curve
        contrast_factor = 1.0 / contrast_boost
        depth_contrast = np.power(depth_normalized, contrast_factor)
        
        return depth_contrast
    
    def apply_gradient_curve(self, depth, power):
        """Apply power curve for better gradients"""
        if power == 1.0:
            return depth
        
        # Apply power curve to enhance gradients
        depth_curved = np.power(depth, power)
        return depth_curved
    
    def final_normalization(self, depth):
        """Final normalization to perfect 0-255 range"""
        # Ensure full dynamic range
        depth_min, depth_max = depth.min(), depth.max()
        
        if depth_max > depth_min:
            # Stretch to full 0-1 range
            normalized = (depth - depth_min) / (depth_max - depth_min)
        else:
            normalized = np.zeros_like(depth)
        
        # Convert to 0-255 uint8
        final_depth = (normalized * 255).astype(np.uint8)
        
        return final_depth

# ===== QUICK CONVERSION FUNCTIONS CELL =====
def convert_to_professional_depth(image, quality="high"):
    """
    One-line function to convert image to professional depth map
    
    Args:
        image: Input image (path or PIL Image)
        quality: "fast", "balanced", "high", "ultra"
    """
    converter = ProfessionalDepthConverter()
    
    # Quality presets
    presets = {
        "fast": {
            "smoothing": 1.0,
            "contrast_boost": 1.3,
            "gradient_power": 1.0,
            "edge_preservation": 0.5
        },
        "balanced": {
            "smoothing": 1.5,
            "contrast_boost": 1.5,
            "gradient_power": 1.1,
            "edge_preservation": 0.7
        },
        "high": {
            "smoothing": 2.0,
            "contrast_boost": 1.7,
            "gradient_power": 1.2,
            "edge_preservation": 0.8
        },
        "ultra": {
            "smoothing": 2.5,
            "contrast_boost": 2.0,
            "gradient_power": 1.3,
            "edge_preservation": 0.9
        }
    }
    
    params = presets.get(quality, presets["high"])
    depth_map = converter.create_professional_depth(image, **params)
    
    return depth_map

def create_smooth_gradient_depth(image, 
                                smoothing=2.0,
                                contrast=1.7, 
                                power=1.2):
    """Create smooth gradient depth map like your reference"""
    converter = ProfessionalDepthConverter()
    return converter.create_professional_depth(
        image, 
        smoothing=smoothing,
        contrast_boost=contrast,
        gradient_power=power,
        edge_preservation=0.8
    )

def compare_depth_styles(image):
    """Compare different depth map styles"""
    converter = ProfessionalDepthConverter()
    
    styles = {
        "Raw": {"smoothing": 0, "contrast_boost": 1.0, "gradient_power": 1.0, "edge_preservation": 0},
        "Smooth": {"smoothing": 3.0, "contrast_boost": 1.2, "gradient_power": 1.0, "edge_preservation": 0.9},
        "High Contrast": {"smoothing": 1.5, "contrast_boost": 2.5, "gradient_power": 1.5, "edge_preservation": 0.7},
        "Professional": {"smoothing": 2.0, "contrast_boost": 1.7, "gradient_power": 1.2, "edge_preservation": 0.8}
    }
    
    results = []
    for style_name, params in styles.items():
        print(f" Creating {style_name} style...")
        depth = converter.create_professional_depth(image, **params)
        results.append((style_name, depth))
    
    # Display comparison
    fig, axes = plt.subplots(1, len(results)+1, figsize=(5*(len(results)+1), 5))
    
    axes[0].imshow(image)
    axes[0].set_title("Original", fontweight='bold')
    axes[0].axis('off')
    
    for i, (name, depth) in enumerate(results):
        if depth is not None:
            axes[i+1].imshow(depth, cmap='gray', vmin=0, vmax=255)
            axes[i+1].set_title(name, fontweight='bold')
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return results

# ===== DISPLAY FUNCTIONS CELL =====
def display_professional_result(original, depth_map, title="Professional Depth Map"):
    """Display original and professional depth map"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    axes[0].imshow(original)
    axes[0].set_title("Original Image", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    if depth_map is not None:
        # Display as grayscale with proper range
        axes[1].imshow(depth_map, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Professional Depth Map\n(Near=Light, Far=Dark)", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def save_professional_depth(depth_map, filename="professional_depth.png"):
    """Save depth map as high-quality image"""
    if depth_map is not None:
        # Convert to PIL Image
        depth_image = Image.fromarray(depth_map, mode='L')
        
        # Save with high quality
        depth_image.save(filename, optimize=True, quality=95)
        print(f" Saved: {filename}")
        
        # Also save as downloadable file in Colab
        try:
            from google.colab import files
            files.download(filename)
        except:
            pass
        
        return depth_image
    return None

# Initialize converter
print(" Initializing professional depth converter...")
depth_converter = ProfessionalDepthConverter()

# ===== UPLOAD AND PROCESS CELL =====
from google.colab import files

print(" Upload your image:")
uploaded = files.upload()

# Load image
image_path = list(uploaded.keys())[0]
original_image = depth_converter.load_image(image_path)

# Display original
plt.figure(figsize=(8, 6))
plt.imshow(original_image)
plt.title("Original Image", fontsize=16, fontweight='bold')
plt.axis('off')
plt.show()

# ===== CREATE PROFESSIONAL DEPTH MAP CELL =====
print(" Creating professional depth map (like your reference)...")

# Create professional depth map
professional_depth = create_smooth_gradient_depth(
    original_image,
    smoothing=2.0,      # Smooth gradients
    contrast=1.7,       # Good contrast
    power=1.2          # Nice curve
)

if professional_depth is not None:
    # Display result
    display_professional_result(original_image, professional_depth)
    
    # Save the result
    save_professional_depth(professional_depth, "my_professional_depth.png")
    
    print(" Professional depth map created successfully!")
    print(" Depth map statistics:")
    print(f"   • Min value: {professional_depth.min()}")
    print(f"   • Max value: {professional_depth.max()}")
    print(f"   • Mean value: {professional_depth.mean():.1f}")
else:
    print(" Failed to create depth map")

# ===== STYLE COMPARISON CELL =====
print("\n Comparing different depth map styles:")
style_results = compare_depth_styles(original_image)

# ===== PARAMETER TUNING CELL =====
print("\n Parameter tuning for perfect results:")

def tune_parameters(image):
    """Interactive parameter tuning"""
    converter = ProfessionalDepthConverter()
    
    # Different parameter combinations
    param_sets = [
        {"smoothing": 1.0, "contrast_boost": 1.5, "gradient_power": 1.0, "name": "Minimal Processing"},
        {"smoothing": 2.0, "contrast_boost": 1.7, "gradient_power": 1.2, "name": "Balanced (Recommended)"},
        {"smoothing": 3.0, "contrast_boost": 2.0, "gradient_power": 1.4, "name": "Maximum Smoothing"},
        {"smoothing": 1.5, "contrast_boost": 2.5, "gradient_power": 1.6, "name": "High Contrast"},
    ]
    
    results = []
    for params in param_sets:
        name = params.pop("name")
        print(f"🔧 Testing: {name}")
        depth = converter.create_professional_depth(image, edge_preservation=0.8, **params)
        results.append((name, depth, params))
    
    # Display parameter comparison
    fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 5))
    if len(results) == 1:
        axes = [axes]
    
    for i, (name, depth, params) in enumerate(results):
        if depth is not None:
            axes[i].imshow(depth, cmap='gray', vmin=0, vmax=255)
        axes[i].set_title(f"{name}", fontweight='bold', fontsize=10)
        axes[i].axis('off')
    
    plt.suptitle("Parameter Comparison", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return results

# Run parameter tuning
tuning_results = tune_parameters(original_image)

# ===== BATCH PROCESSING CELL =====
def batch_convert_images(image_paths, quality="high", output_dir="depth_outputs"):
    """Convert multiple images to professional depth maps"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    converter = ProfessionalDepthConverter()
    results = []
    
    for i, image_path in enumerate(image_paths):
        print(f"\n Processing {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
        
        try:
            # Load image
            image = converter.load_image(image_path)
            
            # Create depth map
            depth_map = convert_to_professional_depth(image, quality)
            
            if depth_map is not None:
                # Save result
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}_depth.png")
                
                depth_image = Image.fromarray(depth_map, mode='L')
                depth_image.save(output_path, optimize=True, quality=95)
                
                results.append({
                    'original': image_path,
                    'depth_map': output_path,
                    'success': True
                })
                
                print(f" Saved: {output_path}")
            else:
                results.append({
                    'original': image_path,
                    'depth_map': None,
                    'success': False
                })
                print(f" Failed: {image_path}")
                
        except Exception as e:
            print(f" Error processing {image_path}: {e}")
            results.append({
                'original': image_path,
                'depth_map': None,
                'success': False
            })
    
    successful = sum(1 for r in results if r['success'])
    print(f"\n Batch processing complete! {successful}/{len(image_paths)} images processed successfully.")
    
    return results

print("\n PROFESSIONAL DEPTH CONVERTER READY!")
print("="*50)
print(" QUICK FUNCTIONS:")
print("• convert_to_professional_depth(image) - One-line conversion")
print("• create_smooth_gradient_depth(image) - Smooth gradients like your reference")
print("• compare_depth_styles(image) - Compare different styles")
print("• batch_convert_images([paths]) - Process multiple images")
print("\n PERFECT FOR:")
print("• Smooth gradient depth maps")
print("• Professional near=light, far=dark visualization")
print("• Clean, noise-free results")
print("• Automotive/photography applications")