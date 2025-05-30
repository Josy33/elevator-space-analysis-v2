"""
Real-Time Elevator Space Analyzer
Integrates depth estimation, occupancy detection, and space analysis for live elevator monitoring
"""

import os
import sys
import cv2
import torch
import numpy as np
import time
import argparse
from datetime import datetime
import threading
import queue
import json

# Add MiDaS to path - adjust this path as needed
midas_path = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\MiDaS-master\\MiDaS-master"
sys.path.append(midas_path)

# Import our modules
from occupancy_detection import OccupancyDetector, SpaceCalculator

class RealTimeSpaceAnalyzer:
    def __init__(self, model_path="data/output/occupancy/best_model.pth", midas_path=None):
        """
        Initialize the real-time space analyzer.
        
        Args:
            model_path (str): Path to trained occupancy model
            midas_path (str): Path to MiDaS directory
        """
        self.model_path = model_path
        self.midas_path = midas_path or "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\MiDaS-master\\MiDaS-master"
        
        # Set up output directories
        self.output_dir = "data/output/realtime"
        self.logs_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print("Loading models...")
        self._load_midas_model()
        self._load_occupancy_model()
        
        # Initialize space calculator
        self.space_calculator = SpaceCalculator()
        
        # Analysis parameters
        self.space_threshold = 30  # Below 30% triggers "FULL" signal
        self.analysis_interval = 1  # Analyze every N seconds
        
        # Statistics tracking
        self.stats = {
            "frames_processed": 0,
            "occupancy_detections": 0,
            "full_signals": 0,
            "average_space": 0,
            "start_time": time.time()
        }
        
        # Logging
        self.log_file = os.path.join(self.logs_dir, f"elevator_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.analysis_log = []
        
    def _load_midas_model(self):
        """Load MiDaS depth estimation model"""
        try:
            sys.path.append(self.midas_path)
            from midas.midas_net_custom import MidasNet_small
            
            model_weights = os.path.join(self.midas_path, "weights", "midas_v21_small-70d6b9c8.pt")
            
            self.midas_model = MidasNet_small(
                model_weights,
                features=64,
                backbone="efficientnet_lite3",
                exportable=True,
                non_negative=True,
                blocks={'expand': True}
            )
            
            self.midas_model = self.midas_model.float()
            self.midas_model.to(self.device)
            self.midas_model.eval()
            
            print("✅ MiDaS depth estimation model loaded")
            
        except Exception as e:
            print(f"❌ Error loading MiDaS model: {e}")
            raise
    
    def _load_occupancy_model(self):
        """Load trained occupancy detection model"""
        try:
            self.occupancy_model = OccupancyDetector(backbone='resnet18', pretrained=False)
            
            if os.path.exists(self.model_path):
                self.occupancy_model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                print("✅ Occupancy detection model loaded")
            else:
                print(f"⚠️  Warning: Model file not found at {self.model_path}")
                print("Using untrained model - results may be inaccurate")
            
            self.occupancy_model.to(self.device)
            self.occupancy_model.eval()
            
        except Exception as e:
            print(f"❌ Error loading occupancy model: {e}")
            raise
    
    def estimate_depth(self, frame):
        """
        Estimate depth from a single frame using MiDaS.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            numpy.ndarray: Depth map (0-255)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_input = cv2.resize(rgb_frame, (256, 256))
        
        # Normalize
        img_input = img_input.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_input = (img_input - mean) / std
        
        # Convert to tensor
        img_tensor = torch.from_numpy(img_input).permute(2, 0, 1).float()
        input_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.midas_model(input_batch)
        
        # Convert to numpy and resize to original frame size
        depth_map = prediction.cpu().numpy().squeeze()
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
        
        # Normalize to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return depth_normalized
    
    def detect_occupancy(self, depth_map):
        """
        Detect occupancy from depth map.
        
        Args:
            depth_map: Grayscale depth map (0-255)
            
        Returns:
            tuple: (is_occupied, probability)
        """
        # Convert to RGB for model input
        rgb_depth = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        rgb_depth = cv2.resize(rgb_depth, (224, 224))
        
        # Apply transforms
        img_tensor = torch.from_numpy(rgb_depth).permute(2, 0, 1).float() / 255.0
        
        # Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        
        input_batch = img_tensor.unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.occupancy_model(input_batch)
            probability = prediction.item()
            is_occupied = probability > 0.5
        
        return is_occupied, probability
    
    def analyze_frame(self, frame):
        """
        Complete analysis of a single frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: Analysis results
        """
        start_time = time.time()
        
        # Step 1: Estimate depth
        depth_map = self.estimate_depth(frame)
        
        # Step 2: Detect occupancy
        is_occupied, occupancy_prob = self.detect_occupancy(depth_map)
        
        # Step 3: Calculate available space
        available_space = self.space_calculator.calculate_space(depth_map)
        
        # Step 4: Make decision
        elevator_status = self._determine_status(available_space, is_occupied)
        
        # Step 5: Create visualization
        visualization = self._create_visualization(frame, depth_map, is_occupied, 
                                                  occupancy_prob, available_space, elevator_status)
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["frames_processed"] += 1
        if is_occupied:
            self.stats["occupancy_detections"] += 1
        if elevator_status == "FULL":
            self.stats["full_signals"] += 1
        
        # Log analysis
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "is_occupied": is_occupied,
            "occupancy_probability": occupancy_prob,
            "available_space_percent": available_space,
            "elevator_status": elevator_status,
            "processing_time_ms": processing_time * 1000
        }
        
        self.analysis_log.append(analysis_result)
        
        return {
            "frame": frame,
            "depth_map": depth_map,
            "visualization": visualization,
            "is_occupied": is_occupied,
            "occupancy_probability": occupancy_prob,
            "available_space": available_space,
            "elevator_status": elevator_status,
            "processing_time": processing_time
        }
    
    def _determine_status(self, available_space, is_occupied):
        """Determine elevator status based on analysis"""
        if not is_occupied:
            return "EMPTY"
        elif available_space < self.space_threshold:
            return "FULL"
        elif available_space < 60:
            return "LIMITED_SPACE"
        else:
            return "AVAILABLE"
    
    def _create_visualization(self, frame, depth_map, is_occupied, prob, space, status):
        """Create analysis visualization"""
        # Create depth visualization
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        # Create space visualization
        space_vis = self.space_calculator.create_visualization(depth_map)
        
        # Combine original frame and visualizations
        h, w = frame.shape[:2]
        
        # Resize all to same height
        frame_resized = cv2.resize(frame, (w//2, h//2))
        depth_resized = cv2.resize(depth_colored, (w//2, h//2))
        space_resized = cv2.resize(space_vis, (w//2, h//2))
        
        # Create combined visualization
        top_row = np.hstack((frame_resized, depth_resized))
        bottom_row = np.hstack((space_resized, np.zeros_like(space_resized)))
        
        # Add status information to bottom right
        info_panel = bottom_row[:, w//2:, :]
        
        # Add text overlays
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Status
        status_color = {
            "EMPTY": (0, 255, 0),
            "AVAILABLE": (0, 255, 0),
            "LIMITED_SPACE": (0, 165, 255),
            "FULL": (0, 0, 255)
        }.get(status, (255, 255, 255))
        
        cv2.putText(info_panel, f"STATUS: {status}", (10, 30), font, 0.6, status_color, 2)
        cv2.putText(info_panel, f"Occupied: {'Yes' if is_occupied else 'No'}", (10, 60), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Confidence: {prob:.2f}", (10, 85), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Space: {space:.1f}%", (10, 110), font, 0.5, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 135), font, 0.4, (255, 255, 255), 1)
        
        # Statistics
        runtime = time.time() - self.stats["start_time"]
        cv2.putText(info_panel, f"Frames: {self.stats['frames_processed']}", (10, 165), font, 0.4, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Runtime: {runtime/60:.1f}m", (10, 185), font, 0.4, (200, 200, 200), 1)
        
        combined = np.vstack((top_row, bottom_row))
        return combined
    
    def run_camera_analysis(self, camera_source=0, display=True, record=False):
        """
        Run real-time analysis on camera feed.
        
        Args:
            camera_source: Camera index or video file path or RTSP URL
            display (bool): Whether to display visualization
            record (bool): Whether to record analysis video
        """
        print(f"Starting camera analysis from source: {camera_source}")
        
        # Open video capture
        cap = cv2.VideoCapture(camera_source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera source: {camera_source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Camera properties: {width}x{height} @ {fps} FPS")
        
        # Set up recording if requested
        out = None
        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"elevator_analysis_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps/2, (width, height*2))  # Adjusted for visualization size
            print(f"🎥 Recording to: {output_path}")
        
        last_analysis_time = 0
        
        try:
            print("🚀 Real-time analysis started. Press 'q' to quit, 's' to save logs.")
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error reading frame or end of video")
                    break
                
                current_time = time.time()
                
                # Analyze frame at specified interval
                if current_time - last_analysis_time >= self.analysis_interval:
                    result = self.analyze_frame(frame)
                    last_analysis_time = current_time
                    
                    # Display results
                    if display:
                        cv2.imshow('Elevator Space Analysis', result['visualization'])
                        
                        # Print status
                        print(f"🔍 Status: {result['elevator_status']} | "
                             f"Space: {result['available_space']:.1f}% | "
                             f"Occupied: {result['is_occupied']} | "
                             f"Processing: {result['processing_time']*1000:.1f}ms")
                    
                    # Record if enabled
                    if record and out is not None:
                        out.write(result['visualization'])
                
                # Check for key presses
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_logs()
                        print("💾 Logs saved!")
                
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted by user")
        
        finally:
            # Clean up
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Save final logs
            self.save_logs()
            self.print_final_stats()
    
    def save_logs(self):
        """Save analysis logs to file"""
        with open(self.log_file, 'w') as f:
            json.dump({
                "analysis_log": self.analysis_log,
                "statistics": self.stats,
                "parameters": {
                    "space_threshold": self.space_threshold,
                    "analysis_interval": self.analysis_interval
                }
            }, f, indent=2)
    
    def print_final_stats(self):
        """Print final statistics"""
        runtime = time.time() - self.stats["start_time"]
        avg_fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
        
        print(f"\n📊 FINAL STATISTICS")
        print(f"   Runtime: {runtime/60:.1f} minutes")
        print(f"   Frames processed: {self.stats['frames_processed']}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Occupancy detections: {self.stats['occupancy_detections']}")
        print(f"   Full signals: {self.stats['full_signals']}")
        print(f"   Logs saved to: {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description='Real-Time Elevator Space Analyzer')
    parser.add_argument('--source', type=str, default=0, 
                       help='Camera source (0 for default camera, path for video file, URL for RTSP)')
    parser.add_argument('--model', type=str, default='data/output/occupancy/best_model.pth',
                       help='Path to trained occupancy model')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--record', action='store_true',
                       help='Record analysis video')
    parser.add_argument('--threshold', type=int, default=30,
                       help='Space threshold for FULL signal (percentage)')
    parser.add_argument('--interval', type=float, default=1.0,
                       help='Analysis interval in seconds')
    
    args = parser.parse_args()
    
    # Handle string input for camera source
    try:
        camera_source = int(args.source)
    except ValueError:
        camera_source = args.source
    
    # Initialize analyzer
    try:
        analyzer = RealTimeSpaceAnalyzer(model_path=args.model)
        analyzer.space_threshold = args.threshold
        analyzer.analysis_interval = args.interval
        
        # Run analysis
        analyzer.run_camera_analysis(
            camera_source=camera_source,
            display=not args.no_display,
            record=args.record
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())