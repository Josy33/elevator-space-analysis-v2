import cv2
import os
import glob
from tqdm import tqdm

class DataCollector:
    def __init__(self, base_path="data"):
        """
        Initialize the data collector.
        
        Args:
            base_path (str): Base path for data storage
        """
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.images_path = os.path.join(self.raw_path, "images")
        self.videos_path = os.path.join(self.raw_path, "videos")
        
        # Create base directories
        for category in ["empty", "occupied", "test"]:
            # Create directories for videos (input)
            video_category_path = os.path.join(self.videos_path, category)
            os.makedirs(video_category_path, exist_ok=True)
            
            # Create directories for extracted frames (output)
            image_category_path = os.path.join(self.images_path, category)
            os.makedirs(image_category_path, exist_ok=True)
            
            print(f"Created directories:")
            print(f"  Videos: {video_category_path}")
            print(f"  Images: {image_category_path}")
    
    def process_video(self, video_path, category="test", frame_interval=30):
        """
        Extract frames from a video file.
        
        Args:
            video_path (str): Path to video file
            category (str): One of ["empty", "occupied", "test"]
            frame_interval (int): Extract every Nth frame
        """
        assert category in ["empty", "occupied", "test"], "Invalid category"
        save_path = os.path.join(self.images_path, category)
        
        # Get video filename without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Processing video: {video_path}")
        print(f"Saving frames to: {save_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video: {video_path}")
            return
            
        # Get video info
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps
        
        print(f"Video info: {frame_count} frames, {fps:.2f} FPS, {duration:.2f} seconds")
        print(f"Extracting every {frame_interval}th frame...")
        
        frames_saved = 0
        frame_idx = 0
        
        try:
            with tqdm(total=frame_count, desc=f"Extracting frames from {video_name}") as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Save every Nth frame
                    if frame_idx % frame_interval == 0:
                        filename = os.path.join(save_path, f"{category}_{video_name}_{frame_idx:06d}.jpg")
                        cv2.imwrite(filename, frame)
                        frames_saved += 1
                    
                    frame_idx += 1
                    pbar.update(1)
                    
        finally:
            cap.release()
        
        print(f"Saved {frames_saved} frames to {save_path}")
        return frames_saved

def collect_video_paths(videos_dir):
    """Collect all video files from the videos directory"""
    videos = []
    
    for category in ["empty", "occupied", "test"]:
        category_dir = os.path.join(videos_dir, category)
        
        if not os.path.exists(category_dir):
            print(f"WARNING: Directory not found: {category_dir}")
            continue
        
        # Find all video files
        video_files = []
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            video_files.extend(glob.glob(os.path.join(category_dir, f"*.{ext}")))
        
        if not video_files:
            print(f"WARNING: No video files found in {category_dir}")
            continue
        
        print(f"Found {len(video_files)} videos in {category_dir}")
        for video_path in video_files:
            videos.append((video_path, category))
    
    return videos

def main():
    print(f"Working directory: {os.getcwd()}")
    # Initialize data collector
    collector = DataCollector(base_path="C:\\Users\\ephra\\Elevator-Analysis-Project\\data")
    
    # Find videos
    videos_dir = os.path.join(collector.raw_path, "videos")
    videos = collect_video_paths(videos_dir)
    
    if not videos:
        print("\nNo videos found to process.")
        print("Please make sure your videos are in the following directories:")
        print(f"  {os.path.join(videos_dir, 'empty')}")
        print(f"  {os.path.join(videos_dir, 'occupied')}")
        print(f"  {os.path.join(videos_dir, 'test')}")
        return
    
    print(f"\nFound {len(videos)} videos to process.")
    
    # Process each video
    total_frames = 0
    for video_path, category in videos:
        frames = collector.process_video(video_path, category)
        if frames:
            total_frames += frames
    
    print(f"\nProcessing complete! Extracted {total_frames} frames total.")
    print(f"Frames are saved in: {collector.images_path}")
    
    # Print next steps
    print("\nNext steps:")
    print("1. Check extracted frames in the directories above")
    print("2. Run augment_data.py to create augmented versions of these frames")
    print("3. Test the depth estimation model with these frames")

if __name__ == "__main__":
    main()