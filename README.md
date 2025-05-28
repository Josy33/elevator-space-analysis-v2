# Elevator Space Analysis

A computer vision system for analyzing elevator space utilization using depth estimation and object detection.

## Project Structure

```
elevator-space-analysis/
├── data/               # Data storage
│   ├── raw/           # Original, immutable data
│   └── processed/     # Cleaned and processed data
├── models/            # Model weights and configurations
│   ├── depth/        # MiDaS depth estimation models
│   └── detection/    # Object detection models
├── src/              # Source code
│   ├── data_pipeline/# Data processing and preparation
│   ├── inference/    # Model inference and analysis
│   └── api/         # API endpoints and services
├── tests/           # Unit and integration tests
├── requirements.txt # Project dependencies
└── README.md       # Project documentation
```

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Models

- Depth Estimation: MiDaS v3
- Object Detection: [TBD]

## Usage

[To be added]

## License

[To be added] 