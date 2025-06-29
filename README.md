# Elevator Space Analyzer
## Real-Time Occupancy Analysis System

---

### Executive Summary
The Elevator Space Analyzer is a computer vision-based system that provides real-time occupancy detection for elevator spaces. The system uses depth estimation and advanced analytics to determine whether an elevator has available space or is at full capacity.

**Key Command:** `python space_analyzer_fixed.py`

---

## System Requirements & Setup

### 1. Installation Prerequisites

Before running the system, ensure you have Python installed and complete the following setup steps:

#### Install Dependencies
```bash
pip install -r requirements.txt
```

#### Download MiDaS Model (Depth Estimation)
```bash
git clone https://github.com/isl-org/MiDaS.git
```

---

## Running the Application

### Testing Mode

For development and testing, use one of the following commands:

```bash
# Using webcam (default camera)
python space_analyzer_fixed.py --source 0

# Using a video file
python space_analyzer_fixed.py --source "video.mp4"
```

### Production Deployment

For production environments with silent operation and recording:

```bash
python space_analyzer_fixed.py --source "rtsp://your-camera-feed" --no-display --record
```

---

## Project Structure

| Component | Description |
|-----------|-------------|
| **space_analyzer_fixed.py** | Main analysis script - this is the primary entry point |
| **data/** | Directory for raw videos and outputs (excluded from version control) |
| **MiDaS/** | Depth estimation model directory |
| **Supporting .py files** | Additional scripts for pre-processing and training |

---

## Implementation Guidelines

### Hardware Compatibility
- **Minimum:** Raspberry Pi 4
- **Recommended:** Standard PC with dedicated GPU for enhanced performance
- **Camera:** Any USB webcam or IP camera with RTSP support

### Input Sources
The system supports multiple input sources:
- **Webcam:** Use `--source 0` (or `1`, `2` for multiple cameras)
- **Video Files:** Use `--source "filename.mp4"`
- **Network Streams:** RTSP URLs from security cameras

### System Outputs
- **Real-time Status:** "FULL" or "AVAILABLE" indicators
- **Logging:** Analysis logs saved to `data/output/`
- **Optional Recording:** Video output with overlay information

---

## Troubleshooting Guide

| Issue | Solution |
|-------|----------|
| **"MiDaS weights not found"** | Verify that the MiDaS repository was cloned successfully |
| **"Can't open camera"** | Check the `--source` argument. Try different values (0, 1, 2) or verify video file path |
| **Slow performance** | Add `--interval 2.0` to reduce analysis frequency |
| **RTSP connection failed** | Verify network connectivity and camera stream URL |

### Performance Optimization
For systems with limited resources:
```bash
python space_analyzer_fixed.py --source 0 --interval 3.0 --low-res
```

---

## Additional Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--source` | Input source (camera index or file path) | 0 |
| `--no-display` | Run without GUI (for headless systems) | False |
| `--record` | Save output video | False |
| `--interval` | Analysis interval in seconds | 1.0 |
| `--threshold` | Occupancy threshold percentage | 80 |

---

## Support & Contact

**Developer:Ridwan Tech team  
**Email:support@ridwantech.com  

### License
This project is licensed under the MIT License - free to use with attribution.

---

## Quick Reference Commands

```bash
# Basic webcam test
python space_analyzer_fixed.py

# Production deployment
python space_analyzer_fixed.py --source "rtsp://192.168.1.100:554/stream" --no-display --record

# Debug mode with verbose output
python space_analyzer_fixed.py --source 0 --verbose --debug
```

---

*Document Version: 1.0*  
*Last Updated: 27/06/2025