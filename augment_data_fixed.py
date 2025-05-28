# augment_data_fixed.py
import cv2
import os
import numpy as np
import glob
from tqdm import tqdm

def augment_image(image):
    """Apply various augmentations to an image"""
    augmented_images = []
    
    # 1. Original image
    augmented_images.append(image.copy())
    
    # 2. Brightness adjustment
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    augmented_images.append(bright)
    
    # 3. Darkness adjustment
    dark = cv2.convertScaleAbs(image, alpha=0.8, beta=-10)
    augmented_images.append(dark)
    
    # 4. Horizontal flip
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 5. Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    augmented_images.append(blurred)
    
    return augmented_images

def process_category(category, input_base, output_base):
    """Process all images in a category"""
    input_dir = os.path.join(input_base, category)
    output_dir = os.path.join(output_base, category)
    
    print(f"\nProcessing category: {category}")
    print(f"Input directory: {os.path.abspath(input_dir)}")
    print(f"Output directory: {os.path.abspath(output_dir)}")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: Input directory does not exist: {input_dir}")
        return 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created/verified: {output_dir}")
    
    # Find all images
    image_files = glob.glob(os.path.join(input_dir, "*.jpg"))
    if not image_files:
        print(f"No images found in {input_dir}")
        return 0
    
    print(f"Found {len(image_files)} images in '{category}' category")
    
    # Process each image
    total_augmented = 0
    for img_path in tqdm(image_files, desc=f"Augmenting {category}"):
        try:
            # Read image
            image = cv2.imread(img_path)
            if image is None:
                print(f"Failed to read image: {img_path}")
                continue
            
            # Get base filename
            filename = os.path.basename(img_path)
            base_name, ext = os.path.splitext(filename)
            
            # Generate augmentations
            augmented_images = augment_image(image)
            
            # Save augmented images
            for i, aug_img in enumerate(augmented_images):
                if i == 0:
                    # Skip the original
                    continue
                    
                aug_filename = f"{base_name}_aug{i}{ext}"
                aug_path = os.path.join(output_dir, aug_filename)
                
                success = cv2.imwrite(aug_path, aug_img)
                if success:
                    total_augmented += 1
                else:
                    print(f"Failed to write image: {aug_path}")
                
                # Verify file was created
                if not os.path.exists(aug_path):
                    print(f"File not found after writing: {aug_path}")
                    
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Verify files in output directory
    output_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    print(f"Created {total_augmented} augmented images for '{category}'")
    print(f"Found {len(output_files)} files in output directory")
    
    return total_augmented

def main():
    # Define directories with absolute paths
    input_base = os.path.join(os.getcwd(), "data", "raw", "images")
    output_base = os.path.join(os.getcwd(), "data", "augmented", "images")
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input base directory: {input_base}")
    print(f"Output base directory: {output_base}")
    
    # Create output base directory
    os.makedirs(output_base, exist_ok=True)
    print(f"Created output base directory: {output_base}")
    
    # Process each category
    categories = ["empty", "occupied", "test"]
    total_augmented = 0
    
    for category in categories:
        category_augmented = process_category(category, input_base, output_base)
        total_augmented += category_augmented
    
    print("\nAugmentation complete!")
    print(f"Total augmented images created: {total_augmented}")
    print(f"Check the output directory: {output_base}")

if __name__ == "__main__":
    main()