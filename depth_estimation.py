"""
Enhanced Depth Estimation with Better Feature Extraction and Checkpoint Resume
Specifically designed to capture floor visibility and occupancy patterns
"""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import gc
from scipy import ndimage

print("===== ENHANCED DEPTH ESTIMATION WITH CHECKPOINT RESUME =====")

# Configuration
MIDAS_PATH = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\Elevator-Analysis-Clean\\MiDaS"
DATA_BASE = "data/augmented/images"
OUTPUT_BASE = "data/output/depth_estimation"

# Create output directory
os.makedirs(OUTPUT_BASE, exist_ok=True)

# Updated categories for 3-class structure
CATEGORIES = ["empty", "partially_occupied", "full", "test"]

# Set up MiDaS
import sys
sys.path.append(MIDAS_PATH)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load model
print("Loading MiDaS model...")
model_type = "MiDaS_small"
model_path = os.path.join(MIDAS_PATH, "weights", "midas_v21_small.pt")

# Check if model file exists
if not os.path.exists(model_path):
    print(f"ERROR: Model file not found at {model_path}")
    print("Please ensure MiDaS weights are downloaded and placed correctly.")
    exit(1)

try:
    # Load model architecture first
    midas = torch.hub.load("intel-isl/MiDaS", model_type, pretrained=False, trust_repo=True)
    
    # Then load the weights
    print(f"Loading weights from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    midas.load_state_dict(checkpoint)
    midas.to(device)
    midas.eval()
    
    # Load transforms
    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.small_transform
    
    print("Model loaded successfully!")
    
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def get_checkpoint_info(output_dir, category):
    """Check which images have already been processed"""
    processed_files = set()
    if os.path.exists(output_dir):
        # Get all existing depth maps
        existing_depth = glob.glob(os.path.join(output_dir, "*_depth.jpg"))
        for depth_file in existing_depth:
            # Extract original filename
            basename = os.path.basename(depth_file)
            # Remove the _aug suffix and _depth suffix to get original name
            original_name = basename.replace("_depth.jpg", ".jpg")
            processed_files.add(original_name)
        
        # Also check for metadata files to ensure complete processing
        existing_meta = glob.glob(os.path.join(output_dir, "*_depth_meta.json"))
        print(f"  Found {len(existing_depth)} depth maps and {len(existing_meta)} metadata files")
    
    return processed_files

def analyze_depth_distribution_enhanced(depth_map):
    """
    Enhanced analysis based on visual observations of elevator occupancy patterns
    """
    try:
        # Normalize to 0-1 for consistent analysis
        depth_norm = depth_map.astype(float) / 255.0
        
        stats = {}
        
        # Basic statistics
        stats['mean_depth'] = float(np.mean(depth_norm))
        stats['std_depth'] = float(np.std(depth_norm))
        stats['min_depth'] = float(np.min(depth_norm))
        stats['max_depth'] = float(np.max(depth_norm))
        
        # 1. FLOOR VISIBILITY ANALYSIS
        # Floor appears as continuous dark regions (low depth values)
        floor_threshold = 0.3  # Dark areas are far/floor
        potential_floor = depth_norm < floor_threshold
        
        # Find largest connected component (likely the floor)
        labeled, num_features = ndimage.label(potential_floor)
        if num_features > 0:
            component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_component_size = max(component_sizes)
            floor_visibility_ratio = largest_component_size / depth_map.size
        else:
            floor_visibility_ratio = 0
            largest_component_size = 0
        
        stats['floor_visibility_ratio'] = float(floor_visibility_ratio)
        stats['floor_continuity'] = float(largest_component_size)
        stats['num_floor_segments'] = int(num_features)
        
        # 2. DEPTH COMPLEXITY/VARIATION
        # Full elevators have more depth variations
        # Calculate entropy as a measure of complexity
        hist, _ = np.histogram(depth_norm, bins=50)
        hist = hist + 1e-10  # Avoid log(0)
        hist_norm = hist / np.sum(hist)
        entropy = -np.sum(hist_norm * np.log(hist_norm))
        stats['depth_entropy'] = float(entropy)
        
        # Local variation (edges/transitions)
        sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        stats['edge_density'] = float(np.mean(edge_magnitude))
        stats['edge_std'] = float(np.std(edge_magnitude))
        
        # 3. DEPTH DISTRIBUTION ANALYSIS
        # Near objects (people) vs far objects (floor/walls)
        stats['near_pixel_ratio'] = float(np.sum(depth_norm > 0.7) / depth_map.size)  # Very bright = very close
        stats['mid_near_ratio'] = float(np.sum((depth_norm > 0.5) & (depth_norm <= 0.7)) / depth_map.size)
        stats['mid_far_ratio'] = float(np.sum((depth_norm > 0.3) & (depth_norm <= 0.5)) / depth_map.size)
        stats['far_pixel_ratio'] = float(np.sum(depth_norm <= 0.3) / depth_map.size)  # Dark = far/floor
        
        # 4. SPATIAL ANALYSIS
        # Vertical profile - in full elevators, depth varies more vertically
        vertical_profile = np.mean(depth_norm, axis=1)
        stats['vertical_variation'] = float(np.std(vertical_profile))
        stats['vertical_range'] = float(np.max(vertical_profile) - np.min(vertical_profile))
        
        # Horizontal profile - people create "peaks" in horizontal scans
        horizontal_profile = np.mean(depth_norm, axis=0)
        stats['horizontal_variation'] = float(np.std(horizontal_profile))
        stats['horizontal_range'] = float(np.max(horizontal_profile) - np.min(horizontal_profile))
        
        # 5. OCCUPANCY-SPECIFIC FEATURES
        # Center vs periphery analysis (people tend to stand in center)
        h, w = depth_map.shape
        center_y, center_x = h // 2, w // 2
        center_region = depth_norm[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
        periphery = np.concatenate([
            depth_norm[:h//4, :].flatten(),  # top
            depth_norm[-h//4:, :].flatten(), # bottom
            depth_norm[:, :w//4].flatten(),  # left
            depth_norm[:, -w//4:].flatten()  # right
        ])
        
        stats['center_mean_depth'] = float(np.mean(center_region))
        stats['periphery_mean_depth'] = float(np.mean(periphery))
        stats['center_periphery_ratio'] = float(stats['center_mean_depth'] / (stats['periphery_mean_depth'] + 1e-10))
        
        # 6. TEXTURE/PATTERN ANALYSIS
        # Calculate local binary patterns or simple texture measure
        # Full elevators have more texture due to clothing, objects, etc.
        texture_window = 5
        texture_scores = []
        for i in range(0, h - texture_window, texture_window):
            for j in range(0, w - texture_window, texture_window):
                window = depth_norm[i:i+texture_window, j:j+texture_window]
                texture_scores.append(np.std(window))
        
        stats['texture_complexity'] = float(np.mean(texture_scores))
        stats['texture_variation'] = float(np.std(texture_scores))
        
        # 7. DEPTH CLUSTERING
        # Number of distinct depth regions (more people = more regions)
        depth_quantized = (depth_norm * 10).astype(int)  # Quantize to 10 levels
        unique_regions = len(np.unique(depth_quantized))
        stats['num_depth_clusters'] = int(unique_regions)
        
        # Most common depth value (mode)
        hist_fine, bins_fine = np.histogram(depth_norm, bins=100)
        mode_idx = np.argmax(hist_fine)
        mode_depth = (bins_fine[mode_idx] + bins_fine[mode_idx + 1]) / 2
        stats['mode_depth'] = float(mode_depth)
        
        return stats
        
    except Exception as e:
        print(f"Error in analyze_depth_distribution_enhanced: {e}")
        # Return basic stats on error
        return {
            'mean_depth': float(np.mean(depth_map)),
            'std_depth': float(np.std(depth_map)),
            'error': str(e)
        }

def enhance_depth_map(depth_map):
    """
    Apply enhancements to make occupancy differences more pronounced
    """
    try:
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(depth_map)
        
        # Apply bilateral filter to reduce noise while preserving edges
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        return enhanced
    except Exception as e:
        print(f"Error in enhance_depth_map: {e}")
        return depth_map

def process_image(image_path, output_path, save_metadata=True):
    """Process a single image with enhanced depth estimation"""
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read image {image_path}")
            return False, None
        
        original_shape = img.shape[:2]
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        input_batch = transform(img).to(device)
        
        # Predict
        with torch.no_grad():
            prediction = midas(input_batch)
            
            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=original_shape,
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy and clear GPU memory if using CUDA
        output = prediction.cpu().numpy()
        del prediction
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Normalize to 0-255
        output_norm = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply enhancements
        output_enhanced = enhance_depth_map(output_norm)
        
        # Analyze depth distribution with enhanced features
        stats = analyze_depth_distribution_enhanced(output_enhanced)
        
        # Save enhanced depth map
        cv2.imwrite(output_path, output_enhanced)
        
        # Save colorized versions
        try:
            # INFERNO colormap - good for general visualization
            output_inferno = cv2.applyColorMap(output_enhanced, cv2.COLORMAP_INFERNO)
            inferno_path = output_path.replace("_depth.jpg", "_depth_inferno.jpg")
            cv2.imwrite(inferno_path, output_inferno)
            
            # JET colormap - good for seeing depth variations
            output_jet = cv2.applyColorMap(output_enhanced, cv2.COLORMAP_JET)
            jet_path = output_path.replace("_depth.jpg", "_depth_jet.jpg")
            cv2.imwrite(jet_path, output_jet)
        except Exception as e:
            print(f"Warning: Could not save color visualizations: {e}")
        
        # Save metadata if requested
        if save_metadata and stats is not None:
            try:
                metadata = {
                    'source_image': os.path.basename(image_path),
                    'timestamp': datetime.now().isoformat(),
                    'original_shape': list(original_shape),
                    'depth_stats': stats
                }
                meta_path = output_path.replace("_depth.jpg", "_depth_meta.json")
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metadata: {e}")
        
        # Clear variables to free memory
        del img, input_batch, output, output_norm
        
        return True, stats
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return False, None

# Process all categories
total_processed = 0
total_skipped = 0
category_stats = {}
failed_images = []

print("\nStarting depth map generation with enhanced features...")
print("Checking for already processed images...")
print("-" * 60)

for category in CATEGORIES:
    input_dir = os.path.join(DATA_BASE, category)
    output_dir = os.path.join(OUTPUT_BASE, category)
    
    if not os.path.exists(input_dir):
        print(f"Skipping {category} - directory not found")
        continue
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check what's already been processed
    already_processed = get_checkpoint_info(output_dir, category)
    
    # Get all images
    all_images = glob.glob(os.path.join(input_dir, "*.jpg"))
    
    # Filter out already processed images
    images_to_process = []
    for img in all_images:
        basename = os.path.basename(img)
        if basename not in already_processed:
            images_to_process.append(img)
    
    print(f"\n{category}:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Already processed: {len(already_processed)}")
    print(f"  Remaining to process: {len(images_to_process)}")
    
    total_skipped += len(already_processed)
    
    if not images_to_process:
        print(f"  ✓ All images already processed!")
        # Load existing stats if available
        stats_path = os.path.join(output_dir, f"{category}_stats.json")
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                category_stats[category] = json.load(f)
        continue
    
    # Process remaining images
    processed = 0
    category_depth_stats = []
    category_failed = []
    
    # If we have some already processed, load their stats
    if len(already_processed) > 0:
        print(f"  Loading stats from already processed images...")
        for meta_file in glob.glob(os.path.join(output_dir, "*_depth_meta.json")):
            try:
                with open(meta_file, 'r') as f:
                    meta_data = json.load(f)
                    if 'depth_stats' in meta_data and 'error' not in meta_data['depth_stats']:
                        category_depth_stats.append(meta_data['depth_stats'])
            except:
                pass
    
    with tqdm(images_to_process, desc=f"Processing {category}") as pbar:
        for idx, img_path in enumerate(pbar):
            filename = os.path.basename(img_path)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{name}_depth.jpg")
            
            success, stats = process_image(img_path, output_path)
            if success:
                processed += 1
                if stats and 'error' not in stats:
                    category_depth_stats.append(stats)
            else:
                category_failed.append(filename)
                failed_images.append(f"{category}/{filename}")
            
            pbar.set_postfix({"processed": processed, "failed": len(category_failed)})
            
            # Periodic garbage collection
            if idx % 50 == 0:
                gc.collect()
    
    print(f"  Newly processed: {processed} images")
    if category_failed:
        print(f"  Failed: {len(category_failed)} images")
    
    total_processed += processed
    
    # Calculate and save category statistics
    if category_depth_stats:
        try:
            # Calculate average stats for the category
            avg_stats = {}
            stat_keys = category_depth_stats[0].keys()
            
            for key in stat_keys:
                values = [s[key] for s in category_depth_stats if key in s]
                if values and all(isinstance(v, (int, float)) for v in values):
                    avg_stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            category_stats[category] = {
                'num_images': len(already_processed) + processed,
                'failed_images': len(category_failed),
                'average_stats': avg_stats
            }
            
            # Save category statistics
            stats_path = os.path.join(output_dir, f"{category}_stats.json")
            with open(stats_path, 'w') as f:
                json.dump(category_stats[category], f, indent=2)
                
        except Exception as e:
            print(f"Error saving category statistics: {e}")

# Save overall statistics
try:
    overall_stats_path = os.path.join(OUTPUT_BASE, "overall_depth_statistics.json")
    with open(overall_stats_path, 'w') as f:
        json.dump({
            'total_processed': total_processed,
            'total_skipped': total_skipped,
            'total_failed': len(failed_images),
            'failed_images': failed_images,
            'categories': category_stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
except Exception as e:
    print(f"Error saving overall statistics: {e}")

print(f"\n===== DEPTH ESTIMATION COMPLETED =====")
print(f"Total newly processed: {total_processed} images")
print(f"Total skipped (already done): {total_skipped} images")
if failed_images:
    print(f"Total failed: {len(failed_images)} images")
print(f"Results saved in: {OUTPUT_BASE}")

# Verify results and show ENHANCED statistics
print("\n" + "="*60)
print("ENHANCED DEPTH MAP ANALYSIS SUMMARY")
print("="*60)

key_metrics = [
    ('floor_visibility_ratio', 'Floor Visibility', 'Higher = more empty'),
    ('edge_density', 'Edge Density', 'Higher = more objects/people'),
    ('near_pixel_ratio', 'Near Objects', 'Higher = more crowded'),
    ('center_periphery_ratio', 'Center Activity', 'Higher = people in center'),
    ('num_depth_clusters', 'Depth Regions', 'More regions = more people'),
    ('texture_complexity', 'Scene Complexity', 'Higher = more variation')
]

for category in CATEGORIES:
    if category in category_stats and 'average_stats' in category_stats[category]:
        print(f"\n{category.upper()}:")
        stats = category_stats[category]['average_stats']
        
        for metric_key, metric_name, description in key_metrics:
            if metric_key in stats:
                mean_val = stats[metric_key]['mean']
                std_val = stats[metric_key]['std']
                print(f"  {metric_name}: {mean_val:.3f} (±{std_val:.3f}) - {description}")

print("\n" + "-"*60)
print("EXPECTED PATTERNS:")
print("- Empty: High floor visibility, low edge density, few depth regions")
print("- Partially Occupied: Moderate values across all metrics")
print("- Full: Low floor visibility, high edge density, many depth regions")
print("-"*60)

print(f"\nDetailed statistics saved to: {overall_stats_path}")
print("\nNext: Run occupancy_detection.py for 3-class training")
print("The enhanced features should now better differentiate between occupancy levels!")