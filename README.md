# AI-Powered Elevator Space Analysis System

A complete computer vision system that analyzes elevator occupancy and calculates available space in real-time using depth estimation and machine learning. The system achieves **99.6% accuracy** in occupancy detection and can process live video feeds or recorded videos.

## 🚀 Key Features

- **Real-time occupancy detection** with 99.6% accuracy
- **Space calculation** showing percentage of available elevator space
- **Multiple input sources**: webcam, IP cameras, RTSP streams, or video files
- **Live visualization** with 4-panel display (original, depth map, space analysis, status)
- **Automated logging** and statistics tracking
- **Configurable thresholds** for "FULL" elevator signals
- **Ready for commercial deployment** in elevator systems

## 📁 Project Structure

```
elevator-space-analysis-v2/
├── data/
│   ├── augmented/
│   │   └── images/          # Augmented training images
│   │       ├── empty/       # Empty elevator images
│   │       ├── occupied/    # Occupied elevator images
│   │       └── test/        # Test images
│   ├── output/
│   │   ├── depth_estimation/# Generated depth maps
│   │   └── occupancy/       # Trained models and results
│   └── raw/
│       ├── images/          # Original extracted frames
│       └── videos/          # Original elevator videos
├── custom_midas.py          # Depth map generation script
├── occupancy_detection.py   # AI model training script
├── space_analyzer.py        # Real-time analysis system
├── process_video.py         # Video frame extraction
├── simple_augment.py        # Data augmentation
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.10 or later
- Git
- At least 4GB RAM (8GB recommended)
- GPU optional but recommended for faster processing

### Step 1: Clone the Repository
```
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download MiDaS (Depth Estimation Model)
```bash
# Clone MiDaS repository
git clone https://github.com/isl-org/MiDaS.git

# Download model weights
cd MiDaS
mkdir -p weights
# Download the small model (recommended for real-time use)
curl -L -o weights/midas_v21_small-70d6b9c8.pt https://github.com/intel-isl/MiDaS/releases/download/v2_1/midas_v21_small-70d6b9c8.pt
cd ..
```

### Step 4: Update Paths in Code
Open the following files and update the MiDaS path to match your installation:

**In `custom_midas.py`:**
```python
midas_path = "path/to/your/MiDaS"  # Update this line
```

**In `space_analyzer.py`:**
```python
midas_path = "path/to/your/MiDaS"  # Update this line
```

## 🎯 Quick Start - Testing the System

### Option 1: Test with Your Own Video File
```bash
python space_analyzer.py --source "path/to/your/elevator_video.mp4"
```

### Option 2: Test with Webcam (if available)
```bash
python space_analyzer.py --source 0
```

### Option 3: Test with Sample Data
If you have the complete dataset, you can train and test the full system:

1. **Extract frames from videos:**
   ```bash
   python process_video.py
   ```

2. **Create augmented training data:**
   ```bash
   python simple_augment.py
   ```

3. **Generate depth maps:**
   ```bash
   python custom_midas.py
   ```

4. **Train the occupancy detection model:**
   ```bash
   python occupancy_detection.py
   ```

5. **Run real-time analysis:**
   ```bash
   python space_analyzer.py
   ```

## 🖥️ Using the Real-Time Analyzer

### Basic Usage
```bash
# Analyze video file
python space_analyzer.py --source "elevator_video.mp4"

# Use webcam
python space_analyzer.py --source 0

# Connect to IP camera (RTSP stream)
python space_analyzer.py --source "rtsp://camera-ip:554/stream"
```

### Advanced Options
```bash
# Record the analysis session
python space_analyzer.py --source "video.mp4" --record

# Change the "FULL" threshold to 25% (default is 30%)
python space_analyzer.py --source "video.mp4" --threshold 25

# Run without display (headless mode)
python space_analyzer.py --source "video.mp4" --no-display

# Analyze every 0.5 seconds instead of 1 second
python space_analyzer.py --source "video.mp4" --interval 0.5
```

### Controls During Analysis
- **Press 'q'** to quit
- **Press 's'** to save current logs
- The system automatically saves logs when you exit

## 📊 Understanding the Output

### Real-Time Display
The system shows a 4-panel view:
1. **Top Left**: Original video feed
2. **Top Right**: Depth map (color-coded distance)
3. **Bottom Left**: Space analysis (red = occupied, other colors = available)
4. **Bottom Right**: Status information and statistics

### Status Indicators
- **🟢 EMPTY**: No one in elevator (100% space available)
- **🟢 AVAILABLE**: People present but plenty of space (>60% available)
- **🟠 LIMITED_SPACE**: Getting crowded (30-60% available)
- **🔴 FULL**: Elevator at capacity (<30% available)

### Log Files
Results are automatically saved to:
- **JSON logs**: `data/output/realtime/logs/elevator_analysis_YYYYMMDD_HHMMSS.json`
- **Analysis videos**: `data/output/realtime/elevator_analysis_YYYYMMDD_HHMMSS.mp4` (if recording enabled)

## 🏢 Commercial Deployment

### Hardware Requirements
- **Computer**: Raspberry Pi 4 or Intel NUC (minimum)
- **Camera**: Existing elevator security cameras or new IP cameras
- **Network**: Ethernet or WiFi connection to building network

### Integration Steps
1. **Install hardware** in elevator machine room
2. **Connect to elevator camera** (RTSP stream or direct connection)
3. **Configure camera URL** in space_analyzer.py
4. **Set appropriate thresholds** for your elevator size
5. **Connect output signals** to elevator controller (GPIO, serial, or network)

### Example Production Command
```bash
# Production deployment with IP camera
python space_analyzer.py \
  --source "rtsp://elevator-cam-01.building.com:554/stream" \
  --threshold 25 \
  --interval 2.0 \
  --no-display \
  --record
```

## 🔧 Troubleshooting

### Common Issues

**"Camera index out of range"**
- Solution: Try different camera indices (0, 1, 2) or use a video file for testing

**"Model weights not found"**
- Solution: Ensure MiDaS weights are downloaded and paths are correctly set

**"ModuleNotFoundError"**
- Solution: Install missing packages with `pip install package_name`

**Poor accuracy results**
- Solution: Ensure the model was trained on similar elevator environments

### Performance Optimization
- **For faster processing**: Use smaller analysis intervals (`--interval 0.5`)
- **For better accuracy**: Use larger models (dpt_large instead of midas_v21_small)
- **For resource-constrained devices**: Increase interval time (`--interval 3.0`)

## 📈 Model Performance

- **Occupancy Detection Accuracy**: 99.6%
- **Training Dataset**: 2,517 depth maps (669 empty + 1,848 occupied)
- **Test Dataset**: 3,393 images
- **Processing Speed**: ~47 frames/second on standard hardware

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For technical support or commercial deployment inquiries:
- Create an issue on GitHub
- Contact: [Your contact information]

## 🎯 Future Enhancements

- [ ] Mobile app for building managers
- [ ] Cloud-based analytics dashboard
- [ ] Multi-elevator building integration
- [ ] Predictive maintenance alerts
- [ ] Integration with building management systems (BMS)

---

**Built with**: Python, PyTorch, OpenCV, MiDaS, and Computer Vision ❤️