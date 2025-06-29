"""
Enhanced Real-Time Elevator Space Analyzer
Integrates depth estimation, 3-class occupancy detection with geometric features
"""

import os
import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
from datetime import datetime
import json
import pickle
from collections import deque
import torchvision.models as models
import torchvision.transforms as transforms
from scipy import ndimage

# Set up paths
MIDAS_PATH = "C:\\Users\\ephra\\OneDrive\\Documents\\Desktop\\Elevator-Analysis-Clean\\MiDaS"
sys.path.append(MIDAS_PATH)

print("===== ENHANCED REAL-TIME ELEVATOR SPACE ANALYZER =====")

class OccupancyDetectorWithFeatures(nn.Module):
    """Same architecture as training - needed for loading"""
    def __init__(self, num_classes=3, num_features=8, use_features=True):
        super().__init__()
        self.use_features = use_features
        
        # Load pre-trained ResNet18
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-20]:
            param.requires_grad = False
        
        # Get the number of features from ResNet
        num_visual_features = self.base_model.fc.in_features
        
        # Remove the final FC layer
        self.base_model.fc = nn.Identity()
        
        if self.use_features:
            # Feature processing branch
            self.feature_branch = nn.Sequential(
                nn.Linear(num_features, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.BatchNorm1d(16),
                nn.ReLU()
            )
            
            # Combined classifier
            self.classifier = nn.Sequential(
                nn.Linear(num_visual_features + 16, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, num_classes)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(num_visual_features, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, x, features=None):
        visual_features = self.base_model(x)
        
        if self.use_features and features is not None:
            geom_features = self.feature_branch(features)
            combined = torch.cat([visual_features, geom_features], dim=1)
            return self.classifier(combined)
        else:
            return self.classifier(visual_features)

class EnhancedRealTimeAnalyzer:
    def __init__(self, model_path="data/output/occupancy/best_model_enhanced.pth"):
        """Initialize the enhanced real-time analyzer"""
        self.model_path = model_path
        
        # Class names
        self.class_names = ['empty', 'partially_occupied', 'full']
        
        # Set up output directories
        self.output_dir = "data/output/realtime"
        self.logs_dir = os.path.join(self.output_dir, "logs")
        self.recordings_dir = os.path.join(self.output_dir, "recordings")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models
        print("\nLoading models...")
        self._load_midas_model()
        self._load_occupancy_model()
        
        # Analysis parameters
        self.analysis_interval = 0.5  # Analyze every 0.5 seconds
        self.alert_threshold = {
            'full': 0.8,  # Alert if confidence > 80% for full
            'partially_occupied': 0.5  # Warning if > 50% for partial
        }
        
        # Space-based correction thresholds
        self.space_correction_enabled = True
        self.space_thresholds = {
            'empty_min': 70,      # > 70% floor = empty
            'full_max': 10        # < 10% floor = full
        }
        
        # Performance tracking and temporal smoothing
        self.fps_buffer = deque(maxlen=30)  # Track last 30 frame times
        self.prediction_history = deque(maxlen=10)  # Track last 10 predictions
        self.current_state = 'partially_occupied'  # Current stable state
        self.state_change_threshold = 5  # Need 5 consistent frames to change state
        
        self.stats = {
            "frames_processed": 0,
            "empty_count": 0,
            "partially_occupied_count": 0,  # Fixed naming
            "full_count": 0,
            "alerts_triggered": 0,
            "start_time": time.time()
        }
        
        # Logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(self.logs_dir, f"elevator_analysis_{timestamp}.json")
        self.analysis_log = []
        
        print("✅ Enhanced Real-Time Analyzer initialized!")
    
    def _load_midas_model(self):
        """Load MiDaS model using the same approach as depth_estimation.py"""
        try:
            # Load model using torch hub
            self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False, trust_repo=True)
            
            # Load weights
            model_path = os.path.join(MIDAS_PATH, "weights", "midas_v21_small.pt")
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                self.midas.load_state_dict(checkpoint)
                print(f"✅ Loaded MiDaS weights from {model_path}")
            else:
                print(f"⚠️  MiDaS weights not found at {model_path}, using default")
            
            self.midas.to(self.device)
            self.midas.eval()
            
            # Load transforms
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.midas_transform = midas_transforms.small_transform
            
            print("✅ MiDaS depth estimation model loaded")
            
        except Exception as e:
            print(f"❌ Error loading MiDaS model: {e}")
            raise
    
    def _load_occupancy_model(self):
        """Load the enhanced 3-class occupancy model with features"""
        try:
            # Load model checkpoint
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Initialize model
            self.occupancy_model = OccupancyDetectorWithFeatures(
                num_classes=3, 
                num_features=8, 
                use_features=True
            )
            
            # Load state dict
            self.occupancy_model.load_state_dict(checkpoint['model_state_dict'])
            self.occupancy_model.to(self.device)
            self.occupancy_model.eval()
            
            # Load feature scaler
            scaler_path = os.path.join("data/output/occupancy", "feature_scaler.pkl")
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                print("✅ Loaded feature scaler")
            else:
                print("⚠️  Feature scaler not found, features may not work correctly")
                self.feature_scaler = None
            
            print(f"✅ Occupancy model loaded (Validation accuracy: {checkpoint.get('best_acc', 'N/A')})")
            
        except Exception as e:
            print(f"❌ Error loading occupancy model: {e}")
            raise
    
    def extract_geometric_features(self, depth_map):
        """Extract the same geometric features used in training"""
        depth_norm = depth_map.astype(float) / 255.0
        
        # Basic features
        features = []
        
        # 1. Edge density (good differentiator)
        sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features.append(float(np.mean(edge_magnitude)))
        
        # 2. Near pixel ratio (excellent differentiator)
        features.append(float(np.sum(depth_norm > 0.7) / depth_map.size))
        
        # 3. Center vs periphery ratio
        h, w = depth_map.shape
        center_y, center_x = h // 2, w // 2
        center_region = depth_norm[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
        periphery = np.concatenate([
            depth_norm[:h//4, :].flatten(),
            depth_norm[-h//4:, :].flatten(),
            depth_norm[:, :w//4].flatten(),
            depth_norm[:, -w//4:].flatten()
        ])
        center_mean = float(np.mean(center_region))
        periphery_mean = float(np.mean(periphery))
        features.append(center_mean / (periphery_mean + 1e-10))
        
        # 4. Far pixel ratio (floor visibility)
        features.append(float(np.sum(depth_norm <= 0.3) / depth_map.size))
        
        # 5. Edge std
        features.append(float(np.std(edge_magnitude)))
        
        # 6-7. Vertical and horizontal variation
        vertical_profile = np.mean(depth_norm, axis=1)
        features.append(float(np.std(vertical_profile)))
        
        horizontal_profile = np.mean(depth_norm, axis=0)
        features.append(float(np.std(horizontal_profile)))
        
        # 8. Texture complexity
        features.append(float(np.std(depth_norm)))
        
        return np.array(features, dtype=np.float32)
    
    def estimate_depth(self, frame):
        """Estimate depth from frame"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply MiDaS transform
        input_batch = self.midas_transform(rgb_frame).to(self.device)
        
        # Run inference
        with torch.no_grad():
            prediction = self.midas(input_batch)
            
            # Resize to original size
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        
        # Normalize to 0-255
        depth_normalized = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply enhancements
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        depth_enhanced = clahe.apply(depth_normalized)
        depth_enhanced = cv2.bilateralFilter(depth_enhanced, 9, 75, 75)
        
        return depth_enhanced
    
    def detect_occupancy(self, depth_map):
        """Detect occupancy using the enhanced model with features"""
        # Extract geometric features
        raw_features = self.extract_geometric_features(depth_map)
        
        # Scale features if scaler available
        if self.feature_scaler:
            features = self.feature_scaler.transform([raw_features])[0]
        else:
            features = raw_features
        
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Prepare image
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Convert depth to RGB
        rgb_depth = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2RGB)
        img_tensor = transform(rgb_depth).unsqueeze(0).to(self.device)
        
        # Run inference with features
        with torch.no_grad():
            outputs = self.occupancy_model(img_tensor, features_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        
        pred_class = predicted.item()
        pred_name = self.class_names[pred_class]
        confidences = probs[0].cpu().numpy()
        
        # Return raw prediction without space correction (will be applied after smoothing)
        return pred_name, confidence.item(), confidences, raw_features
    
    def analyze_frame(self, frame):
        """Complete analysis of a single frame with temporal smoothing then space correction"""
        start_time = time.time()
        
        # Step 1: Estimate depth
        depth_map = self.estimate_depth(frame)
        
        # Step 2: Detect occupancy with features (raw prediction)
        raw_pred_class, raw_confidence, raw_confidences, features = self.detect_occupancy(depth_map)
        
        # Step 3: Calculate available space
        available_space = self.calculate_space_percentage(depth_map)
        
        # Step 4: Apply temporal smoothing on raw predictions
        smoothed_class = self._apply_temporal_smoothing(raw_pred_class, available_space)
        
        # Step 5: Apply space-based correction AFTER smoothing
        final_class = smoothed_class
        final_confidence = raw_confidence
        final_confidences = raw_confidences
        
        if self.space_correction_enabled:
            original_smoothed = smoothed_class
            if available_space < self.space_thresholds['full_max']:
                final_class = 'full'
                final_confidences = np.array([0.02, 0.03, 0.95])
                final_confidence = 0.95
                if original_smoothed != final_class:
                    print(f"   [Space Override: {original_smoothed} → {final_class} based on {available_space:.1f}% floor]")
            elif available_space > self.space_thresholds['empty_min']:
                final_class = 'empty'
                final_confidences = np.array([0.95, 0.03, 0.02])
                final_confidence = 0.95
                if original_smoothed != final_class:
                    print(f"   [Space Override: {original_smoothed} → {final_class} based on {available_space:.1f}% floor]")
        
        # Step 6: Determine alert status
        alert_status = self._determine_alert_status(final_class, final_confidence)
        
        # Step 7: Create visualization
        visualization = self._create_visualization(
            frame, depth_map, final_class, final_confidence, final_confidences, 
            available_space, alert_status, features
        )
        
        processing_time = time.time() - start_time
        self.fps_buffer.append(processing_time)
        
        # Update statistics
        self.stats["frames_processed"] += 1
        self.stats[f"{final_class}_count"] += 1
        if alert_status != "OK":
            self.stats["alerts_triggered"] += 1
        
        # Log analysis
        analysis_result = {
            "timestamp": datetime.now().isoformat(),
            "predicted_class": final_class,
            "raw_prediction": raw_pred_class,
            "smoothed_prediction": smoothed_class,
            "confidence": float(final_confidence),
            "all_confidences": {k: float(v) for k, v in zip(self.class_names, final_confidences)},
            "available_space_percent": float(available_space),
            "alert_status": alert_status,
            "edge_density": float(features[0]),
            "near_pixel_ratio": float(features[1]),
            "processing_time_ms": processing_time * 1000
        }
        
        self.analysis_log.append(analysis_result)
        
        return {
            "frame": frame,
            "depth_map": depth_map,
            "visualization": visualization,
            "predicted_class": final_class,
            "confidence": final_confidence,
            "available_space": available_space,
            "alert_status": alert_status,
            "processing_time": processing_time
        }
    
    def calculate_space_percentage(self, depth_map):
        """Calculate available floor space"""
        depth_norm = depth_map.astype(float) / 255.0
        
        # Floor appears as dark areas
        floor_threshold = 0.3
        floor_mask = depth_norm < floor_threshold
        
        # Find largest connected component (floor)
        from scipy import ndimage
        labeled, num_features = ndimage.label(floor_mask)
        if num_features > 0:
            component_sizes = [np.sum(labeled == i) for i in range(1, num_features + 1)]
            largest_component_size = max(component_sizes)
            floor_visibility_ratio = largest_component_size / depth_map.size
            available_space = floor_visibility_ratio * 100
        else:
            available_space = 0
        
        return available_space
    
    def _determine_alert_status(self, pred_class, confidence):
        """Determine if alert should be triggered"""
        if pred_class == 'full':
            return "ALERT_FULL"  # Always alert for full
        elif pred_class == 'empty':
            return "OK"  # Always OK for empty
        elif pred_class == 'partially_occupied' and confidence > self.alert_threshold['partially_occupied']:
            return "WARNING"
        else:
            return "OK"
    
    def _create_visualization(self, frame, depth_map, pred_class, confidence, 
                            all_confidences, space, alert_status, features):
        """Create comprehensive visualization"""
        # Create depth visualization
        depth_colored = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
        
        # Create panels
        h, w = frame.shape[:2]
        
        # Resize for display
        scale = 0.5
        frame_small = cv2.resize(frame, (int(w*scale), int(h*scale)))
        depth_small = cv2.resize(depth_colored, (int(w*scale), int(h*scale)))
        
        # Create info panel
        info_panel = np.zeros((int(h*scale), int(w*scale), 3), dtype=np.uint8)
        info_panel[:] = (40, 40, 40)  # Dark gray background
        
        # Draw status indicator
        status_color = {
            "OK": (0, 255, 0),
            "WARNING": (0, 165, 255),
            "ALERT_FULL": (0, 0, 255)
        }.get(alert_status, (255, 255, 255))
        
        cv2.rectangle(info_panel, (10, 10), (info_panel.shape[1]-10, 60), status_color, -1)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Status text
        status_text = {
            "empty": "ELEVATOR EMPTY",
            "partially_occupied": "SPACE AVAILABLE",
            "full": "ELEVATOR FULL"
        }.get(pred_class, "UNKNOWN")
        
        cv2.putText(info_panel, status_text, (20, 45), font, 0.8, (255, 255, 255), 2)
        
        # Confidence bars
        y_offset = 80
        for i, (cls, conf) in enumerate(zip(self.class_names, all_confidences)):
            # Draw bar
            bar_width = int(conf * 200)
            bar_color = (0, 255, 0) if cls == pred_class else (100, 100, 100)
            cv2.rectangle(info_panel, (20, y_offset), (20 + bar_width, y_offset + 20), bar_color, -1)
            cv2.rectangle(info_panel, (20, y_offset), (220, y_offset + 20), (200, 200, 200), 1)
            
            # Label
            cv2.putText(info_panel, f"{cls}: {conf:.2%}", (230, y_offset + 15), 
                       font, 0.5, (255, 255, 255), 1)
            y_offset += 30
        
        # Key metrics
        y_offset += 20
        cv2.putText(info_panel, f"Floor Space: {space:.1f}%", (20, y_offset), 
                   font, 0.6, (255, 255, 255), 2)
        
        # Add decision explanation
        y_offset += 25
        if space > 70:
            cv2.putText(info_panel, "Decision: Floor > 70%", (20, y_offset), 
                       font, 0.5, (0, 255, 0), 1)
        elif space < 10:
            cv2.putText(info_panel, "Decision: Floor < 10%", (20, y_offset), 
                       font, 0.5, (0, 0, 255), 1)
        else:
            cv2.putText(info_panel, "Decision: Model + Space", (20, y_offset), 
                       font, 0.5, (200, 200, 200), 1)
        
        y_offset += 25
        cv2.putText(info_panel, f"Edge Density: {features[0]:.2f}", (20, y_offset), 
                   font, 0.5, (200, 200, 200), 1)
        y_offset += 20
        cv2.putText(info_panel, f"Near Objects: {features[1]:.3f}", (20, y_offset), 
                   font, 0.5, (200, 200, 200), 1)
        
        # Performance metrics
        if len(self.fps_buffer) > 0:
            avg_time = np.mean(self.fps_buffer)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            y_offset += 30
            cv2.putText(info_panel, f"FPS: {fps:.1f}", (20, y_offset), 
                       font, 0.5, (150, 150, 150), 1)
        
        # Combine panels
        top_row = np.hstack((frame_small, depth_small))
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(info_panel, timestamp, (20, info_panel.shape[0] - 20), 
                   font, 0.4, (150, 150, 150), 1)
        
        # Stats panel
        stats_panel = np.zeros_like(info_panel)
        stats_panel[:] = (30, 30, 30)
        
        runtime = time.time() - self.stats["start_time"]
        stats_text = [
            f"Runtime: {runtime/60:.1f} min",
            f"Frames: {self.stats['frames_processed']}",
            f"Empty: {self.stats['empty_count']}",
            f"Partial: {self.stats['partially_occupied_count']}",  # Fixed reference
            f"Full: {self.stats['full_count']}",
            f"Alerts: {self.stats['alerts_triggered']}"
        ]
        
        y = 30
        for text in stats_text:
            cv2.putText(stats_panel, text, (20, y), font, 0.5, (200, 200, 200), 1)
            y += 25
        
        bottom_row = np.hstack((info_panel, stats_panel))
        
        # Final combination
        visualization = np.vstack((top_row, bottom_row))
        
        return visualization
    
    def run_analysis(self, source=0, display=True, record=False, max_frames=None):
        """Run real-time analysis on video source"""
        print(f"\n🎥 Starting analysis from source: {source}")
        
        # Handle different source types
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open source: {source}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📹 Video properties: {width}x{height} @ {fps:.1f} FPS")
        
        # Set up recording
        out = None
        if record:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.recordings_dir, f"analysis_{timestamp}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Visualization is 2x2 grid at half scale
            out_width = width
            out_height = height
            out = cv2.VideoWriter(output_path, fourcc, fps/2, (out_width, out_height))
            print(f"🎥 Recording to: {output_path}")
        
        last_analysis_time = 0
        frame_count = 0
        first_frame = True
        
        try:
            print("\n🚀 Analysis started!")
            print("Commands: 'q' to quit, 's' to save logs, 'r' to reset stats")
            print("-" * 60)
            
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Check max frames
                if max_frames and frame_count > max_frames:
                    print(f"\n✅ Reached max frames ({max_frames})")
                    break
                
                current_time = time.time()
                
                # Analyze at specified interval
                if current_time - last_analysis_time >= self.analysis_interval:
                    result = self.analyze_frame(frame)
                    last_analysis_time = current_time
                    
                    # Console output with enhanced display
                    status_icon = {
                        "OK": "✅",
                        "WARNING": "⚠️ ",
                        "ALERT_FULL": "🚨"
                    }.get(result['alert_status'], "❓")
                    
                    # Color-coded output (works in most terminals)
                    class_color = {
                        "empty": "\033[92m",      # Green
                        "partially_occupied": "\033[93m",  # Yellow
                        "full": "\033[91m"        # Red
                    }.get(result['predicted_class'], "\033[0m")
                    reset_color = "\033[0m"
                    
                    # Show confidence and space together for clarity
                    confidence_display = f"{result['confidence']:.0%}"
                    space_display = f"{result['available_space']:.0f}%"
                    
                    # Add clarity indicator for space-based decisions
                    decision_basis = ""
                    if result['available_space'] > 70:
                        decision_basis = " [Floor>70%]"
                    elif result['available_space'] < 10:
                        decision_basis = " [Floor<10%]"
                    
                    print(f"{status_icon} {class_color}{result['predicted_class'].upper():20}{reset_color} | "
                         f"Conf: {confidence_display:5} | "
                         f"Floor: {space_display:4} | "
                         f"Time: {result['processing_time']*1000:5.1f}ms{decision_basis}")
                    
                    # Production Display Note (only show once)
                    if first_frame:
                        first_frame = False
                        print("\n📺 PRODUCTION DISPLAY OPTIONS:")
                        print("   • Simple Display: Just shows GREEN/YELLOW/RED light")
                        print("   • Detailed Display: Shows occupancy + floor space %")
                        print("   • API Integration: JSON data for building systems")
                        print("\n🎯 SIMPLE CLIENT DISPLAY EXAMPLE:")
                        print("   ╔════════════════════════╗")
                        print("   ║   ELEVATOR STATUS      ║")
                        print("   ║                        ║")
                        print("   ║   🟢 ENTER            ║  (Empty: >70% floor)")
                        print("   ║   🟡 LIMITED SPACE    ║  (Partial: 10-70% floor)")
                        print("   ║   🔴 WAIT - FULL      ║  (Full: <10% floor)")
                        print("   ╚════════════════════════╝")
                        print("-" * 60 + "\n")
                    
                    # Display
                    if display:
                        cv2.imshow('Enhanced Elevator Analysis', result['visualization'])
                    
                    # Record
                    if out is not None:
                        out.write(result['visualization'])
                
                # Handle key presses
                if display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self.save_logs()
                        print("💾 Logs saved!")
                    elif key == ord('r'):
                        self._reset_stats()
                        print("🔄 Stats reset!")
                        
        except KeyboardInterrupt:
            print("\n⏹️  Analysis interrupted")
            
        finally:
            # Cleanup
            cap.release()
            if out is not None:
                out.release()
            if display:
                cv2.destroyAllWindows()
            
            # Save final logs
            self.save_logs()
            self.print_summary()
    
    def _apply_temporal_smoothing(self, pred_class, available_space):
        """Apply temporal smoothing based on both prediction and floor space"""
        # Add current prediction to history
        self.prediction_history.append((pred_class, available_space))
        
        # If not enough history, use current prediction
        if len(self.prediction_history) < self.state_change_threshold:
            if len(self.prediction_history) == 1:
                self.current_state = pred_class
            return self.current_state
        
        # Convert deque to list for slicing
        history_list = list(self.prediction_history)
        
        # Analyze recent floor space measurements
        recent_spaces = [space for _, space in history_list[-self.state_change_threshold:]]
        avg_space = np.mean(recent_spaces)
        
        # Determine state based on consistent floor space
        if avg_space > 70 and self.current_state != 'empty':
            # Consistently high floor space = EMPTY
            old_state = self.current_state
            self.current_state = 'empty'
            print(f"   [State Change: {old_state} → empty (avg floor: {avg_space:.1f}%)]")
            return 'empty'
        elif avg_space < 10 and self.current_state != 'full':
            # Consistently low floor space = FULL
            old_state = self.current_state
            self.current_state = 'full'
            print(f"   [State Change: {old_state} → full (avg floor: {avg_space:.1f}%)]")
            return 'full'
        elif 25 <= avg_space <= 55 and self.current_state != 'partially_occupied':
            # Consistently medium floor space = PARTIALLY OCCUPIED
            old_state = self.current_state
            self.current_state = 'partially_occupied'
            print(f"   [State Change: {old_state} → partially_occupied (avg floor: {avg_space:.1f}%)]")
            return 'partially_occupied'
        
        # No state change needed
        return self.current_state
    
    def _reset_stats(self):
        """Reset statistics and temporal smoothing"""
        self.stats = {
            "frames_processed": 0,
            "empty_count": 0,
            "partially_occupied_count": 0,  # Fixed naming
            "full_count": 0,
            "alerts_triggered": 0,
            "start_time": time.time()
        }
        self.analysis_log = []
        self.prediction_history.clear()
        self.current_state = 'partially_occupied'
    
    def save_logs(self):
        """Save analysis logs"""
        log_data = {
            "analysis_log": self.analysis_log,
            "statistics": self.stats,
            "parameters": {
                "analysis_interval": self.analysis_interval,
                "alert_thresholds": self.alert_threshold,
                "model_path": self.model_path
            },
            "runtime": time.time() - self.stats["start_time"]
        }
        
        with open(self.log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def print_summary(self):
        """Print analysis summary"""
        runtime = time.time() - self.stats["start_time"]
        total_frames = self.stats["frames_processed"]
        
        print("\n" + "="*60)
        print("📊 ANALYSIS SUMMARY")
        print("="*60)
        print(f"Runtime: {runtime/60:.1f} minutes")
        print(f"Frames analyzed: {total_frames}")
        
        if total_frames > 0:
            print(f"\nOccupancy Distribution:")
            for cls in self.class_names:
                count = self.stats[f"{cls}_count"]
                percentage = (count / total_frames) * 100
                print(f"  {cls:20}: {count:5} ({percentage:5.1f}%)")
            
            print(f"\nAlerts triggered: {self.stats['alerts_triggered']}")
            
            if len(self.fps_buffer) > 0:
                avg_fps = 1.0 / np.mean(self.fps_buffer)
                print(f"Average FPS: {avg_fps:.1f}")
        
        print(f"\nLogs saved to: {self.log_file}")
        print("\n🚀 PRODUCTION DEPLOYMENT:")
        print("\nSIMPLE CLIENT DISPLAY (What users see):")
        print("┌─────────────────────┐")
        if self.current_state == 'empty':
            print("│    🟢 ENTER        │")
            print("│                    │")
            print("│  Elevator Empty    │")
        elif self.current_state == 'full':
            print("│    🔴 WAIT         │")
            print("│                    │")
            print("│  Elevator Full     │")
        else:
            print("│  🟡 LIMITED SPACE  │")
            print("│                    │")
            print("│  Some Space Left   │")
        print("└─────────────────────┘")
        print("\nThis simple display is all most users need!")
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Enhanced Real-Time Elevator Space Analyzer')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, path for video file)')
    parser.add_argument('--model', type=str, default='data/output/occupancy/best_model_enhanced.pth',
                       help='Path to enhanced model')
    parser.add_argument('--no-display', action='store_true',
                       help='Run without display')
    parser.add_argument('--record', action='store_true',
                       help='Record analysis video')
    parser.add_argument('--interval', type=float, default=0.5,
                       help='Analysis interval in seconds')
    parser.add_argument('--max-frames', type=int, default=None,
                       help='Maximum frames to process')
    parser.add_argument('--empty-threshold', type=int, default=70,
                       help='Floor space % above which elevator is considered empty (default: 70)')
    parser.add_argument('--full-threshold', type=int, default=10,
                       help='Floor space % below which elevator is considered full (default: 10)')
    parser.add_argument('--no-space-correction', action='store_true',
                       help='Disable space-based classification correction')
    parser.add_argument('--smoothing-frames', type=int, default=5,
                       help='Number of consistent frames needed to change state (default: 5)')
    parser.add_argument('--no-smoothing', action='store_true',
                       help='Disable temporal smoothing')
    
    args = parser.parse_args()
    
    try:
        # Initialize analyzer
        analyzer = EnhancedRealTimeAnalyzer(model_path=args.model)
        analyzer.analysis_interval = args.interval
        
        # Configure space correction
        analyzer.space_correction_enabled = not args.no_space_correction
        analyzer.space_thresholds['empty_min'] = args.empty_threshold
        analyzer.space_thresholds['full_max'] = args.full_threshold
        
        # Configure temporal smoothing
        if args.no_smoothing:
            analyzer.state_change_threshold = 1  # No smoothing
        else:
            analyzer.state_change_threshold = args.smoothing_frames
        
        if analyzer.space_correction_enabled:
            print(f"\n📏 Space-based correction enabled:")
            print(f"   Empty threshold: >{args.empty_threshold}% floor space")
            print(f"   Full threshold: <{args.full_threshold}% floor space")
        
        if not args.no_smoothing:
            print(f"\n🔄 Temporal smoothing enabled:")
            print(f"   State change requires {args.smoothing_frames} consistent frames")
        
        # Run analysis
        analyzer.run_analysis(
            source=args.source,
            display=not args.no_display,
            record=args.record,
            max_frames=args.max_frames
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())