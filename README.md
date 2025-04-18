# Pedestrian Detection System

## Overview
The Pedestrian Detection System is a state-of-the-art computer vision application that utilizes deep learning models for real-time pedestrian detection and tracking. This system provides robust pedestrian detection capabilities with multiple interface options including desktop application, web interface, and command-line tools.

## Features

### Detection & Tracking
- Real-time pedestrian detection using YOLOv8 and YOLOv10 models
- Multi-object tracking with consistent ID assignment
- Configurable confidence thresholds for detection
- GPU acceleration support for improved performance
- Explainable AI (XAI) visualizations to understand model decisions

### User Interfaces
- **Desktop Application**: Full-featured GUI with video playback controls
- **Web Interface**: Browser-based access with real-time updates
- **Streamlit App**: Interactive dashboard for analytics
- **Command Line**: Scriptable interface for batch processing

### Analytics & Reporting
- Real-time statistics on pedestrian counts
- Heatmap generation for pedestrian movement patterns
- Zone-based monitoring capabilities
- Performance metrics tracking (FPS, inference time)
- Dwell time analysis

## System Architecture

The system is built with a modular architecture consisting of several key components:

1. **Object Detector**: Handles real-time pedestrian detection using YOLO models
2. **Tracker Module**: Tracks detected pedestrians across video frames
3. **Analytics Module**: Generates insights from detection and tracking data
4. **Visualization Module**: Renders detection results and analytics
5. **Web/Desktop Interfaces**: Provides user interaction capabilities

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- Git

### Setup

1. Clone the repository
```bash
git clone https://github.com/username/pedestrian-detection.git
cd pedestrian-detection
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Download pre-trained models
```bash
python download_models.py
```
This will download the YOLOv10 and other required model weights.

## Usage

### Desktop Application

Run the desktop application for a full-featured GUI experience:

```bash
python desktop_app.py
```

### Command Line Interface

Process a video file with the command line interface:

```bash
python main.py --input videos/test.mp4 --output outputs/result.mp4 --model yolov10x.pt --conf 0.3
```

### Web Interface

Start the web server for browser-based access:

```bash
python website/app.py
```
Then open your browser and navigate to `http://localhost:5000`

### Streamlit Dashboard

Launch the Streamlit analytics dashboard:

```bash
cd streamlit
streamlit run streamlit_app.py
```

## Configuration

The system can be configured through the `configs/config.yaml` file. Key configuration options include:

- Detection model selection (YOLOv8, YOLOv10)
- Confidence thresholds
- Tracking parameters
- Analytics settings
- Visualization options

## Performance

- **Detection Accuracy**: Achieves 90% accuracy in pedestrian detection across various lighting conditions and scenarios
- **Processing Speed**: Maintains real-time processing at 26 FPS on recommended hardware configurations

## Project Structure

```
├── configs/            # Configuration files
├── docs/               # Documentation files
├── models/             # Pre-trained model weights
├── modules/            # Core system modules
│   ├── detector.py     # Object detection module
│   ├── tracker.py      # Object tracking module
│   ├── analytics.py    # Analytics generation
│   └── xai.py          # Explainable AI visualizations
├── outputs/            # Output videos and results
├── performance metrics/# Performance evaluation tools
├── streamlit/          # Streamlit dashboard app
├── utils/              # Utility functions
├── videos/             # Sample videos for testing
├── website/            # Web interface files
├── desktop_app.py      # Desktop application
└── main.py             # Command line interface
```

## Dependencies

The system relies on the following key libraries:

- PyTorch and TorchVision for deep learning
- OpenCV for image processing
- Ultralytics for YOLO implementation
- Flask for web interface
- Streamlit for analytics dashboard
- Captum for XAI visualizations

See `requirements.txt` for a complete list of dependencies.

## License

[MIT License](LICENSE)

## Acknowledgements

- YOLOv8 and YOLOv10 model architectures
- SORT tracking algorithm
- Ultralytics YOLO implementation