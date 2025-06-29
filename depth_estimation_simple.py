"""
Enhanced Depth Estimation for 3-Class Elevator Occupancy Detection
Generates depth maps optimized for distinguishing Empty, Partially Occupied, and Full states
"""

import os
import glob
import cv2
import torch
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime
import gc  # for garbage collection

print("===== ENHANCED DEPTH ESTIMATION FOR 3-CLASS DETECTION =====")

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

# Load model - Fixed approach
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

def analyze_depth_distribution(depth_map):
    """
    Analyze depth distribution to help distinguish between occupancy levels
    """
    try:
        # Calculate statistics that might help differentiate classes
        stats = {
            'mean_depth': float(np.mean(depth_map)),
            'std_depth': float(np.std(depth_map)),
            'min_depth': float(np.min(depth_map)),
            'max_depth': float(np.max(depth_map)),
            'depth_range': float(np.max(depth_map) - np.min(depth_map))
        }
        
        # Calculate depth histogram percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            stats[f'percentile_{p}'] = float(np.percentile(depth_map, p))
        
        # Calculate "near object" ratio
        near_threshold = np.percentile(depth_map, 75)
        near_ratio = np.sum(depth_map > near_threshold) / depth_map.size
        stats['near_object_ratio'] = float(near_ratio)
        
        # Calculate "floor visibility"
        floor_threshold = np.percentile(depth_map, 25)
        floor_ratio = np.sum(depth_map < floor_threshold) / depth_map.size
        stats['floor_visibility_ratio'] = float(floor_ratio)
        
        return stats
    except Exception as e:
        print(f"Error in analyze_depth_distribution: {e}")
        return None

def enhance_depth_map(depth_map):
    """
    Apply enhancements to make occupancy differences more pronounced
    """
    try:
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(depth_map)
        
        # Apply slight gaussian blur
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
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
        
        # Analyze depth distribution
        stats = analyze_depth_distribution(output_enhanced)
        
        # Save enhanced depth map
        cv2.imwrite(output_path, output_enhanced)
        
        # Save colorized versions
        try:
            # INFERNO colormap
            output_inferno = cv2.applyColorMap(output_enhanced, cv2.COLORMAP_INFERNO)
            inferno_path = output_path.replace("_depth.jpg", "_depth_inferno.jpg")
            cv2.imwrite(inferno_path, output_inferno)
            
            # JET colormap
            output_jet = cv2.applyColorMap(output_enhanced, cv2.COLORMAP_JET)
            jet_path = output_path.replace("_depth.jpg", "_depth_jet.jpg")
            cv2.imwrite(jet_path, output_jet)
        except Exception as e:
            print(f"Warning: Could not save color visualizations: {e}")
        
        # Save metadata if requested and stats were calculated successfully
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
category_stats = {}
failed_images = []

for category in CATEGORIES:
    input_dir = os.path.join(DATA_BASE, category)
    output_dir = os.path.join(OUTPUT_BASE, category)
    
    if not os.path.exists(input_dir):
        print(f"Skipping {category} - directory not found")
        continue
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all images
    images = glob.glob(os.path.join(input_dir, "*.jpg"))
    print(f"\nProcessing {category}: {len(images)} images")
    
    if not images:
        continue
    
    # Process with progress bar
    processed = 0
    category_depth_stats = []
    category_failed = []
    
    with tqdm(images, desc=f"Processing {category}") as pbar:
        for idx, img_path in enumerate(pbar):
            filename = os.path.basename(img_path)
            name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{name}_depth.jpg")
            
            success, stats = process_image(img_path, output_path)
            if success:
                processed += 1
                if stats:
                    category_depth_stats.append(stats)
            else:
                category_failed.append(filename)
                failed_images.append(f"{category}/{filename}")
            
            pbar.set_postfix({"success": processed, "failed": len(category_failed)})
            
            # Periodic garbage collection to free memory (especially on CPU)
            if idx % 50 == 0:
                gc.collect()
    
    print(f"Processed {processed}/{len(images)} images in {category}")
    if category_failed:
        print(f"Failed images: {len(category_failed)}")
    
    total_processed += processed
    
    # Calculate and save category statistics
    if category_depth_stats:
        try:
            # Calculate average stats for the category
            avg_stats = {}
            for key in category_depth_stats[0].keys():
                values = [s[key] for s in category_depth_stats if key in s]
                if values:
                    avg_stats[key] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
            
            category_stats[category] = {
                'num_images': processed,
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
            'total_failed': len(failed_images),
            'failed_images': failed_images,
            'categories': category_stats,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
except Exception as e:
    print(f"Error saving overall statistics: {e}")

print(f"\n===== DEPTH ESTIMATION COMPLETED =====")
print(f"Total processed: {total_processed} images")
if failed_images:
    print(f"Total failed: {len(failed_images)} images")
print(f"Results saved in: {OUTPUT_BASE}")

# Verify results and show statistics
print("\nDepth Map Generation Summary:")
for category in CATEGORIES:
    try:
        depth_files = glob.glob(os.path.join(OUTPUT_BASE, category, "*_depth.jpg"))
        inferno_files = glob.glob(os.path.join(OUTPUT_BASE, category, "*_depth_inferno.jpg"))
        jet_files = glob.glob(os.path.join(OUTPUT_BASE, category, "*_depth_jet.jpg"))
        meta_files = glob.glob(os.path.join(OUTPUT_BASE, category, "*_depth_meta.json"))
        
        print(f"\n{category}:")
        print(f"  - Depth maps: {len(depth_files)}")
        print(f"  - Visualizations: {len(inferno_files)} (inferno), {len(jet_files)} (jet)")
        print(f"  - Metadata files: {len(meta_files)}")
        
        # Show key statistics
        if category in category_stats and 'average_stats' in category_stats[category]:
            stats = category_stats[category]['average_stats']
            if 'floor_visibility_ratio' in stats:
                print(f"  - Avg floor visibility: {stats['floor_visibility_ratio']['mean']:.3f}")
            if 'near_object_ratio' in stats:
                print(f"  - Avg near object ratio: {stats['near_object_ratio']['mean']:.3f}")
    except Exception as e:
        print(f"Error showing statistics for {category}: {e}")

print(f"\nOverall statistics saved to: {overall_stats_path}")
print("\nNext: Run occupancy_detection.py for 3-class training")