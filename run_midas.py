from src.data_pipeline.data_collection import DataCollector

def main():
    """
    Run data collection process.

    Usage exapmles:
    1. Captures from webcam for empty elevator:
       - Uncomment the first block
    2. Capture from webcam for occupied elevator:
       - Change category to "occupied" in the first block
    3. Process a video file:
       - Uncomment the second block and provide video path
    """

     # Initialize data collector 
    collector = DataCollector()
    
    # Option 1: Process a video file
    video_path = "path/to/your/elevator_video.mp4"
    print(f"Processing video: {video_path}")
    collector.process_video(video_path, category="test")
    print("Video processing complete!")

    # Option 2: Collect from webcam
    # print("Starting webcam capture for 30 seconds...")
    # print("Press 'q' to stop early")
    # collector.capture_from_camera(duration_seconds=30, category="test")
    # print("Webcam capture complete!")
    

if __name__ == "__main__":
    main()
    