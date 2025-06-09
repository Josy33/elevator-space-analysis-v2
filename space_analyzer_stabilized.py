"""
Real-Time Elevator Space Analyzer with Capacity Stabilization
Implements hysteresis buffer and rolling average to prevent relay switching issues
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
from collections import deque

# Add MiDaS to path - adjust this path as needed
midas_path = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\MiDaS-master\\MiDaS-master"
sys.path.append(midas_path)

# Import our modules
from occupancy_detection import OccupancyDetector, SpaceCalculator

class CapacityStabilizer:
    """
    Stabilization logic to prevent capacity percentage fluctuations
    Implements hysteresis buffer and rolling average as described in project plan
    """
    def __init__(self, buffer_size=5, full_threshold=30, recovery_threshold=60):
        """
        Initialize capacity stabilizer
        
        Args:
            buffer_size (int): Number of frames for rolling average
            full_threshold (int): Percentage threshold to trigger FULL (30% space remaining = 70% full)
            recovery_threshold (int): Percentage threshold to exit FULL (60% space remaining = 40% full)
        """
        self.buffer = deque(maxlen=buffer_size)
        self.is_full = False
        self.full_threshold = full_threshold  # 30% space remaining
        self.recovery_threshold = recovery_threshold  # 60% space remaining
        self.lock_time = None
        self.min_lock_duration = 2.0  # Minimum seconds to stay in FULL state
        
        print(f"🔒 Capacity Stabilizer initialized:")
        print(f"   Full threshold: {full_threshold}% space remaining")
        print(f"   Recovery threshold: {recovery_threshold}% space remaining") 
        print(f"   Buffer size: {buffer_size} frames")
    
    def update(self, current_space_percentage):
        """
        Update stabilizer with new space percentage reading
        
        Args:
            current_space_percentage (float): Current available space percentage (0-100)
            
        Returns:
            tuple: (is_full_status, stabilized_percentage, status_locked)
        """
        # Add to rolling buffer
        self.buffer.append(current_space_percentage)
        
        # Calculate stabilized reading (rolling average)
        if len(self.buffer) >= 3:
            stabilized_space = np.mean(list(self.buffer))
        else:
            stabilized_space = current_space_percentage
        
        current_time = time.time()
        status_locked = False
        
        # Hysteresis logic to prevent oscillation
        if not self.is_full:
            # Check if we should enter FULL state
            # Require 3 out of last 5 readings to be below threshold
            if len(self.buffer) >= 3:
                low_readings = sum(1 for space in self.buffer if space <= self.full_threshold)
                if low_readings >= 3:  # 3 out of 5 frames confirm FULL
                    self.is_full = True
                    self.lock_time = current_time
                    print(f"🔴 ENTERING FULL STATE - Space: {stabilized_space:.1f}%")
        else:
            # Currently in FULL state - check if we can exit
            time_locked = current_time - self.lock_time if self.lock_time else 0
            
            if time_locked < self.min_lock_duration:
                # Still in minimum lock period
                status_locked = True
            else:
                # Check if we can exit FULL (hysteresis - higher threshold to exit)
                if stabilized_space >= self.recovery_threshold:
                    # Additional confirmation - require sustained recovery
                    high_readings = sum(1 for space in self.buffer if space >= self.recovery_threshold)
                    if high_readings >= 4:  # 4 out of 5 frames confirm recovery
                        self.is_full = False
                        self.lock_time = None
                        print(f"🟢 EXITING FULL STATE - Space: {stabilized_space:.1f}%")
                    else:
                        status_locked = True  # Still confirming recovery
                else:
                    status_locked = True  # Stay locked, space still low
        
        return self.is_full, stabilized_space, status_locked
    
    def get_status_text(self, is_full, stabilized_space, status_locked):
        """Get human-readable status text"""
        if is_full:
            return "FULL" + (" 🔒" if status_locked else "")
        elif stabilized_space < 60:
            return "LIMITED_SPACE"
        else:
            return "AVAILABLE"

class RealTimeSpaceAnalyzer:
    def __init__(self, model_path="data/output/occupancy/best_model.pth", midas_path=None):
        """
        Initialize the real-time space analyzer with stabilization.
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
        
        # Initialize capacity stabilizer with project requirements
        self.stabilizer = CapacityStabilizer(
            buffer_size=5,           # 5-frame rolling average
            full_threshold=30,       # FULL when ≤30% space remaining
            recovery_threshold=60    # Exit FULL when ≥60% space remaining
        )
        
        # Analysis parameters
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds for responsiveness
        
        # Statistics tracking
        self.stats = {
            "frames_processed": 0,
            "occupancy_detections": 0,
            "full_signals": 0,
            "false_triggers_prevented": 0,
            "stabilization_events": 0,
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
        """Estimate depth from a single frame using MiDaS."""
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
        """Detect occupancy from depth map."""
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
        """Complete analysis of a single frame with stabilization."""
        start_time = time.time()
        
        # Step 1: Estimate depth
        depth_map = self.estimate_depth(frame)
        
        # Step 2: Detect occupancy
        is_occupied, occupancy_prob = self.detect_occupancy(depth_map)
        
        # Step 3: Calculate raw available space
        raw_space = self.space_calculator.calculate_space(depth_map)
        
        # Step 4: Apply stabilization logic (KEY IMPROVEMENT)
        is_full, stabilized_space, status_locked = self.stabilizer.update(raw_space)
        
        # Step 5: Get stabilized status
        elevator_status = self.stabilizer.get_status_text(is_full, stabilized_space, status_locked)
        
        # Step 6: Create visualization
        visualization = self._create_stabilized_visualization(
            frame, depth_map, is_occupied, occupancy_prob, 
            raw_space, stabilized_space, elevator_status, status_locked, is_full
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats["frames_processed"] += 1
        if is_occupied:
            self.stats["occupancy_detections"] += 1
        if is_full:
            self.stats["full_signals"] += 1
        if status_locked:
            self.stats["stabilization_events"] += 1
        
        # Log analysis
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "is_occupied": is_occupied,
            "occupancy_probability": occupancy_prob,
            "raw_space_percent": raw_space,
            "stabilized_space_percent": stabilized_space,
            "elevator_status": elevator_status,
            "is_full": is_full,
            "status_locked": status_locked,
            "processing_time_ms": processing_time * 1000
        }
        
        self.analysis_log.append(analysis_result)
        
        return {
            "frame": frame,
            "depth_map": depth_map,
            "visualization": visualization,
            "is_occupied": is_occupied,
            "occupancy_probability": occupancy_prob,
            "raw_space": raw_space,
            "stabilized_space": stabilized_space,
            "elevator_status": elevator_status,
            "is_full": is_full,
            "status_locked": status_locked,
            "processing_time": processing_time
        }
    
    def _create_stabilized_visualization(self, frame, depth_map, is_occupied, prob, 
                                       raw_space, stabilized_space, status, status_locked, is_full):
        """Create analysis visualization with stabilization indicators"""
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
        
        # Status with enhanced indicators
        if is_full:
            status_color = (0, 0, 255)  # Red for FULL
        elif stabilized_space < 60:
            status_color = (0, 165, 255)  # Orange for LIMITED
        else:
            status_color = (0, 255, 0)  # Green for AVAILABLE
        
        # Main status display
        cv2.putText(info_panel, f"STATUS: {status}", (10, 30), font, 0.6, status_color, 2)
        
        # Space information with raw vs stabilized
        cv2.putText(info_panel, f"Raw Space: {raw_space:.1f}%", (10, 55), font, 0.4, (200, 200, 200), 1)
        cv2.putText(info_panel, f"Stable Space: {stabilized_space:.1f}%", (10, 75), font, 0.5, (255, 255, 255), 2)
        
        # Occupancy info
        cv2.putText(info_panel, f"Occupied: {'Yes' if is_occupied else 'No'}", (10, 100), font, 0.4, (255, 255, 255), 1)
        cv2.putText(info_panel, f"Confidence: {prob:.2f}", (10, 120), font, 0.4, (255, 255, 255), 1)
        
        # Stabilization indicators
        if status_locked:
            cv2.putText(info_panel, "🔒 LOCKED", (10, 145), font, 0.5, (255, 255, 0), 2)
        
        # Buffer status
        buffer_size = len(self.stabilizer.buffer)
        cv2.putText(info_panel, f"Buffer: {buffer_size}/5", (10, 170), font, 0.4, (200, 200, 200), 1)
        
        cv2.putText(info_panel, f"Time: {datetime.now().strftime('%H:%M:%S')}", (10, 195), font, 0.4, (255, 255, 255), 1)
        
        # Statistics
        cv2.putText(info_panel, f"Frames: {self.stats['frames_processed']}", (10, 220), font, 0.3, (150, 150, 150), 1)
        cv2.putText(info_panel, f"Locks: {self.stats['stabilization_events']}", (10, 235), font, 0.3, (150, 150, 150), 1)
        
        combined = np.vstack((top_row, bottom_row))
        return combined
    
    def run_camera_analysis(self, camera_source=0, display=True, record=False):
        """Run real-time analysis on camera feed with enhanced stabilization."""
        print(f"Starting stabilized camera analysis from source: {camera_source}")
        
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
            output_path = os.path.join(self.output_dir, f"stabilized_analysis_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps/2, (width, height*2))
            print(f"🎥 Recording to: {output_path}")
        
        last_analysis_time = 0
        
        try:
            print("🚀 Stabilized analysis started. Press 'q' to quit, 's' to save logs.")
            print("🔒 Hysteresis buffer active - prevents capacity fluctuations")
            
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
                        cv2.imshow('Stabilized Elevator Space Analysis', result['visualization'])
                        
                        # Enhanced status display
                        lock_icon = "🔒" if result['status_locked'] else "🔓"
                        full_icon = "🔴" if result['is_full'] else "🟢"
                        
                        print(f"{full_icon} {lock_icon} Status: {result['elevator_status']} | "
                             f"Raw: {result['raw_space']:.1f}% | "
                             f"Stable: {result['stabilized_space']:.1f}% | "
                             f"Buffer: {len(self.stabilizer.buffer)}/5")
                    
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
                "stabilization_parameters": {
                    "full_threshold": self.stabilizer.full_threshold,
                    "recovery_threshold": self.stabilizer.recovery_threshold,
                    "buffer_size": self.stabilizer.buffer.maxlen,
                    "min_lock_duration": self.stabilizer.min_lock_duration,
                    "analysis_interval": self.analysis_interval
                }
            }, f, indent=2)
    
    def print_final_stats(self):
        """Print final statistics with stabilization metrics"""
        runtime = time.time() - self.stats["start_time"]
        avg_fps = self.stats["frames_processed"] / runtime if runtime > 0 else 0
        
        print(f"\n📊 STABILIZED ANALYSIS STATISTICS")
        print(f"   Runtime: {runtime/60:.1f} minutes")
        print(f"   Frames processed: {self.stats['frames_processed']}")
        print(f"   Average FPS: {avg_fps:.1f}")
        print(f"   Full signals: {self.stats['full_signals']}")
        print(f"   Stabilization events: {self.stats['stabilization_events']}")
        print(f"   False triggers prevented: Hysteresis buffer active")
        print(f"   Logs saved to: {self.log_file}")

def main():
    parser = argparse.ArgumentParser(description='Stabilized Real-Time Elevator Space Analyzer')
    parser.add_argument('--source', type=str, default="C:\\Users\\ephra\\Downloads\\test 15.mp4", 
                       help='Camera source (0 for default camera, path for video file, URL for RTSP)')
    parser.add_argument('--model', type=str, default='data/output/occupancy/best_model.pth',
                       help='Path to trained occupancy model')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    parser.add_argument('--record', action='store_true',
                       help='Record analysis video')
    
    args = parser.parse_args()
    
    # Handle string input for camera source
    try:
        camera_source = int(args.source)
    except ValueError:
        camera_source = args.source
    
    # Initialize analyzer
    try:
        analyzer = RealTimeSpaceAnalyzer(model_path=args.model)
        
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