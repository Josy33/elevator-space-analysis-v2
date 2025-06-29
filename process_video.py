import cv2
import os
import glob
from tqdm import tqdm

class DataCollector:
    def __init__(self, base_path="data"):
        """Initialize the data collector."""
        self.base_path = base_path
        self.raw_path = os.path.join(base_path, "raw")
        self.images_path = os.path.join(self.raw_path, "images")
        self.videos_path = os.path.join(self.raw_path, "videos")
        
        # Create directories for new 3-class structure
        self.categories = ["empty", "partially_occupied", "full", "test"]
        for category in self.categories:
            os.makedirs(os.path.join(self.videos_path, category), exist_ok=True)
            os.makedirs(os.path.join(self.images_path, category), exist_ok=True)
    
    def process_video(self, video_path, category="test", frame_interval=30):
        """Extract frames from a video file."""
        save_path = os.path.join(self.images_path, category)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"\nProcessing: {os.path.basename(video_path)}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Could not open video")
            return 0
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  Frames: {frame_count}, FPS: {fps:.1f}")
        print(f"  Extracting every {frame_interval}th frame...")
        
        frames_saved = 0
        frame_idx = 0
        
        try:
            with tqdm(total=frame_count, desc="Extracting", leave=False) as pbar:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    if frame_idx % frame_interval == 0:
                        filename = os.path.join(save_path, f"{category}_{video_name}_{frame_idx:06d}.jpg")
                        cv2.imwrite(filename, frame)
                        frames_saved += 1
                    
                    frame_idx += 1
                    pbar.update(1)
                    
        finally:
            cap.release()
        
        print(f"  Saved: {frames_saved} frames")
        return frames_saved

def main():
    print("=== FRAME EXTRACTION FROM VIDEOS (3-CLASS) ===\n")
    
    collector = DataCollector(base_path="data")
    videos_dir = os.path.join(collector.raw_path, "videos")
    
    # Collect all videos from new categories
    all_videos = []
    category_stats = {}
    
    for category in collector.categories:
        category_dir = os.path.join(videos_dir, category)
        video_files = glob.glob(os.path.join(category_dir, "*.mp4"))
        video_files.extend(glob.glob(os.path.join(category_dir, "*.MP4")))
        video_files.extend(glob.glob(os.path.join(category_dir, "*.mov")))
        video_files.extend(glob.glob(os.path.join(category_dir, "*.MOV")))
        
        category_stats[category] = len(video_files)
        print(f"{category}: {len(video_files)} videos")
        for video in video_files:
            all_videos.append((video, category))
    
    if not all_videos:
        print("\nNo videos found!")
        print("Make sure videos are in:")
        for category in collector.categories:
            print(f"  - {os.path.join(videos_dir, category)}/")
        return
    
    print(f"\nTotal videos to process: {len(all_videos)}")
    print("-" * 40)
    
    # Process all videos
    total_frames = 0
    frames_by_category = {cat: 0 for cat in collector.categories}
    
    for video_path, category in all_videos:
        frames = collector.process_video(video_path, category)
        total_frames += frames
        frames_by_category[category] += frames
    
    print("\n" + "=" * 40)
    print(f"COMPLETE! Extracted {total_frames} frames total")
    print("\nFrames per category:")
    for category, count in frames_by_category.items():
        print(f"  {category}: {count} frames")
    print(f"\nLocation: {collector.images_path}")
    print("\nNext: Run simple_augment.py")

if __name__ == "__main__":
    main()