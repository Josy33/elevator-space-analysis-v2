import cv2
import os
import numpy as np
import glob
from tqdm import tqdm

def augment_image(image, category):
    """Apply various augmentations to an image based on category"""
    augmented_images = []
    
    # 1. Original image
    augmented_images.append(image.copy())
    
    # 2. Brightness variations (important for different elevator lighting)
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=20)
    augmented_images.append(bright)
    
    dark = cv2.convertScaleAbs(image, alpha=0.7, beta=-20)
    augmented_images.append(dark)
    
    # 3. Horizontal flip (still valid - elevator layout can be mirrored)
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)
    
    # 4. Motion blur (common in elevators with moving people)
    size = 15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    motion_blurred = cv2.filter2D(image, -1, kernel_motion_blur)
    augmented_images.append(motion_blurred)
    
    # 5. Gaussian noise (camera sensor noise)
    noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
    noisy = cv2.add(image, noise)
    augmented_images.append(noisy)
    
    # 6. Contrast adjustment (different camera settings)
    contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    augmented_images.append(contrast)
    
    # Category-specific augmentations
    if category in ["partially_occupied", "full"]:
        # 7. Extra motion blur for occupied elevators (people movement)
        heavy_motion = cv2.GaussianBlur(image, (9, 9), 0)
        augmented_images.append(heavy_motion)
    
    return augmented_images

def calculate_augmentation_factor(category_counts):
    """Calculate how many augmentations needed per image to balance classes"""
    max_count = max(category_counts.values())
    augmentation_factors = {}
    
    for category, count in category_counts.items():
        if count > 0:
            # Calculate factor to reach approximately the same number as max class
            factor = min(8, max(1, int(max_count / count)))  # Cap at 8x augmentation
            augmentation_factors[category] = factor
        else:
            augmentation_factors[category] = 1
    
    return augmentation_factors

def process_category(category, input_base, output_base, augmentation_factor=None):
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
            augmented_images = augment_image(image, category)
            
            # Select augmentations based on factor
            if augmentation_factor and augmentation_factor > 1:
                # Use more augmentations for underrepresented classes
                selected_augmentations = augmented_images[:augmentation_factor]
            else:
                selected_augmentations = augmented_images
            
            # Save augmented images
            for i, aug_img in enumerate(selected_augmentations):
                aug_filename = f"{base_name}_aug{i}{ext}"
                aug_path = os.path.join(output_dir, aug_filename)
                
                success = cv2.imwrite(aug_path, aug_img)
                if success:
                    total_augmented += 1
                else:
                    print(f"Failed to write image: {aug_path}")
                    
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    # Verify files in output directory
    output_files = glob.glob(os.path.join(output_dir, "*.jpg"))
    print(f"Created {total_augmented} augmented images for '{category}'")
    print(f"Actual files in output: {len(output_files)}")
    
    return total_augmented

def main():
    # Define directories
    input_base = os.path.join(os.getcwd(), "data", "raw", "images")
    output_base = os.path.join(os.getcwd(), "data", "augmented", "images")
    
    print("=== DATA AUGMENTATION FOR 3-CLASS ELEVATOR DETECTION ===\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Input base directory: {input_base}")
    print(f"Output base directory: {output_base}")
    
    # Create output base directory
    os.makedirs(output_base, exist_ok=True)
    
    # Updated categories for 3-class structure
    categories = ["empty", "partially_occupied", "full", "test"]
    
    # First, count images in each category
    category_counts = {}
    print("\nCounting images per category...")
    for category in categories:
        input_dir = os.path.join(input_base, category)
        if os.path.exists(input_dir):
            count = len(glob.glob(os.path.join(input_dir, "*.jpg")))
            category_counts[category] = count
            print(f"  {category}: {count} images")
        else:
            category_counts[category] = 0
            print(f"  {category}: directory not found")
    
    # Calculate augmentation factors for class balancing
    augmentation_factors = calculate_augmentation_factor(category_counts)
    print("\nAugmentation factors for balancing:")
    for cat, factor in augmentation_factors.items():
        print(f"  {cat}: {factor}x augmentation")
    
    # Process each category
    total_augmented = 0
    category_results = {}
    
    for category in categories:
        factor = augmentation_factors.get(category, 1)
        category_augmented = process_category(category, input_base, output_base, factor)
        total_augmented += category_augmented
        category_results[category] = category_augmented
    
    print("\n" + "="*50)
    print("AUGMENTATION COMPLETE!")
    print("="*50)
    print(f"\nTotal augmented images created: {total_augmented}")
    print("\nBreakdown by category:")
    for category, count in category_results.items():
        print(f"  {category}: {count} images")
    print(f"\nAll augmented images saved to: {output_base}")
    print("\nNext: Run occupancy_detection.py for training")

if __name__ == "__main__":
    main() 