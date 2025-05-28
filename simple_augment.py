# simple_augment.py
import cv2
import os
import glob

def main():
    print("Starting simple augmentation...")
    
    # Define paths
    source_base = "data/raw/images"
    dest_base = "data/augmented/images"
    
    # Process each category
    for category in ["empty", "occupied", "test"]:
        source_dir = os.path.join(source_base, category)
        dest_dir = os.path.join(dest_base, category)
        
        print(f"\nProcessing category: {category}")
        print(f"Source: {os.path.abspath(source_dir)}")
        print(f"Destination: {os.path.abspath(dest_dir)}")
        
        # Check source directory
        if not os.path.exists(source_dir):
            print(f"Source directory doesn't exist: {source_dir}")
            continue
            
        # Create destination directory
        os.makedirs(dest_dir, exist_ok=True)
        
        # Get all images
        images = glob.glob(os.path.join(source_dir, "*.jpg"))
        print(f"Found {len(images)} images")
        
        # Process images
        count = 0
        for img_path in images:
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Failed to read: {img_path}")
                    continue
                
                # Get filename
                filename = os.path.basename(img_path)
                name, ext = os.path.splitext(filename)
                
                # Create augmentations
                # 1. Brightness up
                bright = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
                bright_path = os.path.join(dest_dir, f"{name}_bright{ext}")
                cv2.imwrite(bright_path, bright)
                
                # 2. Brightness down
                dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-30)  
                dark_path = os.path.join(dest_dir, f"{name}_dark{ext}")
                cv2.imwrite(dark_path, dark)
                
                # 3. Flip horizontal
                flip = cv2.flip(img, 1)
                flip_path = os.path.join(dest_dir, f"{name}_flip{ext}")
                cv2.imwrite(flip_path, flip)
                
                count += 3
                
                # Print progress every 10 images
                if (images.index(img_path) + 1) % 10 == 0:
                    print(f"Processed {images.index(img_path) + 1}/{len(images)} images")
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        print(f"Created {count} augmented images for {category}")
    
    print("\nAugmentation complete!")

if __name__ == "__main__":
    main()