import cv2
from src.inference.depth_estimation import DepthEstimator
# Later we'll import other components like:
# from src.inference.object_detection import ObjectDetector
# from src.api.server import app

def main():
    """
    Main entry point for the elevator space analysis system.
    """
    # Initialize models
    depth_estimator = DepthEstimator(model_type="DPT_Large")
    
    # Example: Process a single image
    image_path = "data/raw/example.jpg"  # You'll need to add your image
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
            
        # Get depth map
        depth_map = depth_estimator.estimate_depth(image)
        
        # Save processed depth map
        cv2.imwrite("data/processed/depth_map.png", depth_map)
        
        print("Successfully processed image and generated depth map!")
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 