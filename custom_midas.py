"""
Custom MiDaS Depth Estimation for Elevator Space Analysis
This script uses MiDaS's internal libraries but doesn't rely on run.py
"""

import os
import sys
import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm

print("===== CUSTOM MIDAS DEPTH ESTIMATION =====")

# Set paths
midas_path = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\MiDaS-master\\MiDaS-master"
data_base = "data/augmented/images"  # Or "data/raw/images"
output_base = "data/output/depth_estimation"

# Create output directory
os.makedirs(output_base, exist_ok=True)

print(f"MiDaS path: {midas_path}")
print(f"Data path: {data_base}")
print(f"Output path: {output_base}")

# Add MiDaS to path
sys.path.append(midas_path)

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Import MiDaS components directly
    print("Importing MiDaS components...")
    from midas.midas_net_custom import MidasNet_small

    # Load model
    print("Loading MiDaS small model...")
    model_path = os.path.join(midas_path, "weights", "midas_v21_small-70d6b9c8.pt")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at: {model_path}")
    
    print(f"Model weights found at: {model_path}")
    
    # Initialize model
    model = MidasNet_small(
        model_path,
        features=64,
        backbone="efficientnet_lite3",
        exportable=True,
        non_negative=True,
        blocks={'expand': True}
    )
    
    # Convert model to float
    model = model.float()
    
    # Move model to device
    model.to(device)
    model.eval()
    print("Model loaded successfully!")

    # Function to process a single image
    def process_image(img_path, out_path):
        print(f"Processing: {img_path}")
        
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return False
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to 256x256
        img_input = cv2.resize(img, (256, 256))
        
        # Convert to float and normalize
        img_input = img_input.astype(np.float32) / 255.0
        
        # Normalize using ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = (img_input - mean) / std
        
        # HWC -> CHW tensor format
        img_input = torch.from_numpy(img_input).permute(2, 0, 1)
        
        # Make sure input is float32 (to match model)
        img_input = img_input.float()
        
        # Add batch dimension
        input_batch = img_input.unsqueeze(0).to(device)
        
        # Run inference
        with torch.no_grad():
            prediction = model(input_batch)
        
        # Convert to numpy and resize to original image dimensions
        prediction = prediction.cpu().numpy().squeeze()
        prediction = cv2.resize(prediction, (img.shape[1], img.shape[0]))
        
        # Normalize to 0-255 for visualization
        depth_normalized = cv2.normalize(prediction, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Create colored depth map
        depth_colored = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Save depth map and visualization
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Save grayscale depth map
        depth_path = os.path.join(out_path, f"{base_name}_depth.jpg")
        cv2.imwrite(depth_path, depth_normalized)
        
        # Save colored depth map
        depth_color_path = os.path.join(out_path, f"{base_name}_depth_vis.jpg")
        cv2.imwrite(depth_color_path, depth_colored)
        
        return True
    
    # Process images by category
    total_processed = 0
    
    for category in ["empty", "occupied", "test"]:
        category_dir = os.path.join(data_base, category)
        if not os.path.exists(category_dir):
            print(f"Directory not found: {category_dir}")
            continue
        
        # Create output directory for this category
        category_output = os.path.join(output_base, category)
        os.makedirs(category_output, exist_ok=True)
        
        # Get all images
        images = glob.glob(os.path.join(category_dir, "*.jpg"))
        print(f"\nProcessing {len(images)} images in {category} category")
        
        if not images:
            continue
        
        # Process images with progress bar
        successful = 0
        with tqdm(total=len(images), desc=f"Processing {category}", unit="img") as pbar:
            for img_path in images:
                if process_image(img_path, category_output):
                    successful += 1
                pbar.update(1)
                
                # Update progress bar with success rate
                pbar.set_postfix({"success": f"{successful}/{pbar.n}"})
        
        print(f"Successfully processed {successful}/{len(images)} images in {category}")
        total_processed += successful
    
    print(f"\nDepth estimation complete! Processed {total_processed} images total.")
    print(f"Results saved in: {output_base}")
    
    # Check results
    print("\nVerifying results:")
    depth_maps = glob.glob(os.path.join(output_base, "**", "*_depth.jpg"), recursive=True)
    print(f"Found {len(depth_maps)} depth maps")
    
    if depth_maps:
        print("Sample results:")
        for result in depth_maps[:5]:
            print(f"  {result}")
        if len(depth_maps) > 5:
            print(f"  ... and {len(depth_maps) - 5} more")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("===== CUSTOM MIDAS DEPTH ESTIMATION COMPLETED =====")