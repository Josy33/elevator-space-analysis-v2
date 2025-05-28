
"""
MiDaS Direct Runner - Uses MiDaS's run.py script directly to generate depth maps
"""

import os
import sys
import glob
import shutil
import subprocess
from tqdm import tqdm

print("===== MIDAS DIRECT RUNNER =====")

# Set paths
midas_path = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\MiDaS-master\\MiDaS-master"
data_base = "data/augmented/images"  # Or "data/raw/images"
output_base = "data/output/depth_estimation"

# Create output directory
os.makedirs(output_base, exist_ok=True)

print(f"MiDaS path: {midas_path}")
print(f"Data base path: {data_base}")
print(f"Output path: {output_base}")

# Check if MiDaS path exists
if not os.path.exists(midas_path):
    print(f"Error: MiDaS path does not exist: {midas_path}")
    sys.exit(1)

# Check if run.py exists in MiDaS path
run_py_path = os.path.join(midas_path, "run.py")
if not os.path.exists(run_py_path):
    print(f"Error: run.py not found at {run_py_path}")
    sys.exit(1)
else:
    print(f"Found run.py at {run_py_path}")

# Process categories
categories = ["empty", "occupied", "test"]
total_processed = 0

for category in categories:
    category_dir = os.path.join(data_base, category)
    if not os.path.exists(category_dir):
        print(f"Directory not found: {category_dir}")
        continue
        
    print(f"\nProcessing category: {category}")
    
    # Create output directories
    category_output = os.path.join(output_base, category)
    os.makedirs(category_output, exist_ok=True)
    
    # Get all images
    images = glob.glob(os.path.join(category_dir, "*.jpg"))
    print(f"Found {len(images)} images in {category_dir}")
    
    if not images:
        continue
    
    # Create a temporary directory for MiDaS input/output
    temp_input = os.path.join("temp_midas", "input")
    temp_output = os.path.join("temp_midas", "output")
    os.makedirs(temp_input, exist_ok=True)
    os.makedirs(temp_output, exist_ok=True)
    
    # Process images in batches of 10 to avoid memory issues
    batch_size = 10
    for i in range(0, len(images), batch_size):
        batch = images[i:i+batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(images)-1)//batch_size + 1} ({len(batch)} images)")
        
        # Clear temporary directories
        for file in glob.glob(os.path.join(temp_input, "*")):
            os.remove(file)
        for file in glob.glob(os.path.join(temp_output, "*")):
            os.remove(file)
        
        # Copy batch to temporary input directory
        for img_path in batch:
            filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(temp_input, filename))
            print(f"Copied {img_path} to {os.path.join(temp_input, filename)}")
        
        # Run MiDaS using its own run.py
        cmd = f"cd {midas_path} && python run.py --model_type small --input_path {os.path.abspath(temp_input)} --output_path {os.path.abspath(temp_output)}"
        print(f"Executing command: {cmd}")
        
        # Use subprocess.run to capture output and errors
        try:
            result = subprocess.run(
                cmd, 
                shell=True,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Command output:\n{result.stdout}")
            if result.stderr:
                print(f"Command errors:\n{result.stderr}")
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error code {e.returncode}")
            print(f"Error output:\n{e.stderr}")
            continue
        
        # Move results to category output directory
        result_count = 0
        for img_path in batch:
            filename = os.path.basename(img_path)
            name, ext = os.path.splitext(filename)
            
            # MiDaS output files
            depth_png = os.path.join(temp_output, f"{name}.png")
            depth_pfm = os.path.join(temp_output, f"{name}.pfm")
            
            # Check if results exist
            if os.path.exists(depth_png):
                # Copy to category output
                output_path = os.path.join(category_output, f"{name}_depth.png")
                shutil.copy(depth_png, output_path)
                print(f"Copied depth map to {output_path}")
                result_count += 1
            else:
                print(f"Warning: Expected output file not found: {depth_png}")
            
            if os.path.exists(depth_pfm):
                # Copy to category output if needed
                output_path = os.path.join(category_output, f"{name}_depth.pfm")
                shutil.copy(depth_pfm, output_path)
                print(f"Copied PFM file to {output_path}")
        
        print(f"Successfully processed {result_count}/{len(batch)} images in this batch")
        total_processed += result_count
    
    print(f"Completed processing {category} category")

# Clean up temporary directory
try:
    shutil.rmtree("temp_midas", ignore_errors=True)
    print("Cleaned up temporary directory")
except Exception as e:
    print(f"Warning: Error cleaning up temporary directory: {e}")

print(f"\nDepth estimation complete! Processed {total_processed} images total.")
print(f"Results saved in: {output_base}")

# Check results
print("\nVerifying results:")
all_results = glob.glob(os.path.join(output_base, "**", "*_depth.png"), recursive=True)
print(f"Found {len(all_results)} depth maps")

if all_results:
    print("Sample results:")
    for result in all_results[:5]:
        print(f"  {result}")
    if len(all_results) > 5:
        print(f"  ... and {len(all_results) - 5} more")

print("===== MIDAS DIRECT RUNNER COMPLETED =====")