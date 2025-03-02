# Pedestrian Detection and Traffic Management System

A computer vision system that provides both standalone pedestrian detection and a full traffic management interface. The project consists of two main components:
1. Command-line pedestrian detection tool
2. Streamlit web interface for traffic management

## Features

### Pedestrian Detection (main.py)
- Real-time pedestrian detection using YOLOv10
- Interactive control panel with keyboard shortcuts
- Live detection count
- Recording capabilities
- Adjustable visualization options

### Traffic Management System (streamlit_app.py)
- Multi-class detection (vehicles, pedestrians)
- Traffic signal control
- Vehicle density analysis
- Emergency vehicle detection
- Real-time analytics dashboard

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pedestrian-detection
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Usage

### Pedestrian Detection (Command Line)
```bash
python main.py --input <video_path> [--output output.avi] [--log-level INFO]
```
#### Controls:
- R: Start/Stop recording
- B: Toggle bounding boxes
- L: Toggle labels
- Space: Pause/Resume
- Q: Quit

### Traffic Management System (Web Interface)
```bash
streamlit run streamlit_app.py
```
## Project Structure
```project/
├── modules/
│ ├── init.py
│ ├── detector.py # Object detection module
│ ├── traffic_analyzer.py # Traffic analysis
│ ├── traffic_signal_controller.py # Signal control
│ └── emergency_vehicle_handler.py # Emergency detection
├── configs/
│ └── config.yaml # Configuration settings
├── main.py # CLI pedestrian detection
├── streamlit_app.py # Web interface
└── requirements.txt # Dependencies
```

## Requirements
- Python 3.8+
- OpenCV
- PyTorch
- Ultralytics YOLOv10
- Streamlit
- NumPy
- Other dependencies in requirements.txt

## Configuration

The system can be configured through `configs/config.yaml`:
- Detection confidence thresholds
- Model paths
- Traffic analysis parameters
- Signal timing settings

## Model

The system uses YOLOv10 for object detection:
- Default model: YOLOv10
- Supported classes: persons, vehicles
- Configurable confidence threshold

## Output

### Pedestrian Detection
- Processed video with detections
- Detection count
- Optional recording output

### Traffic Management
- Real-time analytics
- Traffic density maps
- Signal status
- Vehicle counts
- Performance metrics


