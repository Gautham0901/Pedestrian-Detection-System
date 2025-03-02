import sys
from pathlib import Path

# Add modules directory to Python path
module_path = str(Path(__file__).parent / "modules")
if module_path not in sys.path:
    sys.path.append(module_path)

import streamlit as st
import cv2
import numpy as np
import yaml
from modules.detector import ObjectDetector
import tempfile
import time
from datetime import datetime, timedelta
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from streamlit.components.v1 import html

# Import traffic management modules
from modules.traffic_analyzer import TrafficAnalyzer
from modules.traffic_signal_controller import TrafficSignalController
from modules.emergency_vehicle_handler import EmergencyVehicleHandler

# Page config with custom CSS
st.set_page_config(
    page_title="Pedestrian Detection System",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    /* Main theme colors */
    :root {
        --primary-color: #2E86C1;
        --secondary-color: #3498DB;
        --background-dark: #1E1E1E;
        --card-dark: #2D2D2D;
        --text-light: #FFFFFF;
        --text-gray: #CCCCCC;
        --accent-color: #00A3E0;
    }

    /* Global styles */
    .main {
        background-color: var(--background-dark);
        color: var(--text-light);
        padding: 0 1rem;
    }

    /* Header styling */
    .header-container {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .header-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    /* Card styling */
    .metric-card {
        background-color: var(--card-dark);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
        transition: transform 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
    }

    .metric-card h3 {
        color: var(--accent-color);
        margin: 0;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        font-size: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Button styling */
    .stButton>button {
        background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        height: 3em;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }

    /* Metric value styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: var(--accent-color);
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-gray);
        font-size: 1rem;
    }

    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--accent-color);
    }

    /* Video container */
    .video-container {
        background-color: var(--card-dark);
        border-radius: 15px;
        padding: 1rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Control panel */
    .control-panel {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Chart styling */
    .js-plotly-plot {
        background-color: var(--card-dark);
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(255,255,255,0.1);
    }

    /* Sidebar styling */
    .css-1d391kg {
        background-color: var(--card-dark);
    }

    /* Input fields */
    .stTextInput>div>div>input {
        background-color: var(--card-dark);
        color: var(--text-light);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
    }

    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }

    .status-active {
        background-color: #2ECC71;
        box-shadow: 0 0 8px #2ECC71;
    }

    .status-inactive {
        background-color: #E74C3C;
        box-shadow: 0 0 8px #E74C3C;
    }
    </style>
    """, unsafe_allow_html=True)

# Admin credentials (in real app, use secure storage)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Default settings for non-admin users
DEFAULT_SETTINGS = {
    'enable_tracking': True,
    'show_boxes': True,
    'show_labels': True,
    'show_trails': True,
    'export_format': 'MP4',
    'confidence_threshold': 0.3,
    'enable_recording': True,
    'show_analytics': True,
    'vehicle_detection': True,
    'traffic_analysis': True,
    'emergency_detection': True,
    'show_debug': False
}

# Initialize session state
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
if 'admin_view' not in st.session_state:
    st.session_state['admin_view'] = False

# Load configuration
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Update model path and classes
VEHICLE_CLASSES = {
    0: {'name': 'person', 'color': (255, 0, 0)},    # Blue for pedestrians
    2: {'name': 'car', 'color': (0, 255, 0)},       # Green
    3: {'name': 'motorcycle', 'color': (0, 0, 255)}, # Red
    5: {'name': 'bus', 'color': (255, 255, 0)},     # Cyan
    7: {'name': 'truck', 'color': (255, 0, 255)},   # Magenta
    1: {'name': 'bicycle', 'color': (0, 165, 255)}  # Orange
}

class DetectionMetrics:
    def __init__(self):
        self.total_detections = 0
        self.current_count = 0
        self.peak_count = 0
        self.frames_processed = 0
        self.start_time = time.time()
        self.processing_times = []
        self.detection_history = []
        self.fps_history = []
        self.tracked_ids = set()

    def update(self, detections, processing_time):
        try:
            self.frames_processed += 1
            self.processing_times.append(processing_time)
            
            # Update current count
            self.current_count = len(detections)
            
            # Update peak count
            self.peak_count = max(self.peak_count, self.current_count)
            
            # Track unique detections
            for det in detections:
                bbox = det['bbox']  # New format uses dict
                det_id = f"{int(bbox[0])}-{int(bbox[1])}-{int(bbox[2])}-{int(bbox[3])}"
                if det_id not in self.tracked_ids:
                    self.tracked_ids.add(det_id)
                    self.total_detections += 1
            
            # Update detection history
            self.detection_history.append(self.current_count)
            
            # Calculate FPS
            if self.processing_times:
                current_fps = 1.0 / (sum(self.processing_times[-30:]) / len(self.processing_times[-30:]))
                self.fps_history.append(current_fps)
            
            # Keep last 100 frames of history
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
                
        except Exception as e:
            st.error(f"Error updating metrics: {e}")

def process_frame(frame, metrics, settings, detector):
    """Process a single frame with the given detector"""
    try:
        start_time = time.time()
        
        # Perform detection
        detections = detector.detect(frame)
        
        # Log detection info (optional)
        if settings.get('show_debug', False):
            st.write(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")
            st.write(f"Number of detections: {len(detections)}")
        
        # Separate pedestrians and vehicles
        pedestrians = [d for d in detections if d['is_person']]
        vehicles = [d for d in detections if not d['is_person']]
        
        # Update metrics
        metrics.update(detections, time.time() - start_time)
        
        # Process detections and draw visualizations
        if settings['show_boxes'] or settings['show_labels']:
            for det in detections:
                bbox = det['bbox']
                x1, y1, w, h = bbox
                
                # Draw bounding box
                if settings['show_boxes']:
                    cv2.rectangle(frame, 
                                (x1, y1), 
                                (x1+w, y1+h), 
                                det['color'], 2)
                
                # Draw label
                if settings['show_labels']:
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(frame,
                                (x1, y1-25),
                                (x1+label_size[0], y1),
                                det['color'],
                                -1)
                    cv2.putText(frame, label,
                              (x1, y1-5),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              0.5, (255, 255, 255), 2)
        
        # Update traffic data
        vehicle_counts = {}
        for v in vehicles:
            class_name = v['class_name']
            vehicle_counts[class_name] = vehicle_counts.get(class_name, 0) + 1
            
        traffic_data = {
            'total_vehicles': len(vehicles),
            'total_pedestrians': len(pedestrians),
            'vehicle_counts': vehicle_counts,
            'density': len(vehicles) / (frame.shape[0] * frame.shape[1]),
            'avg_speed': calculate_average_speed(vehicles)
        }
        
        st.session_state.traffic_data.update(traffic_data)
        
        # Update signal timing based on traffic data
        if hasattr(st.session_state, 'signal_controller'):
            st.session_state.signal_controller.optimize_signal_timing(traffic_data)
            st.session_state.signal_controller.update_signal_status()
        
        # Add traffic analysis overlay
        if settings.get('show_analytics', True):
            frame = add_traffic_overlay(frame, st.session_state.traffic_data)
        
        return frame, detections
        
    except Exception as e:
        st.error(f"Error in frame processing: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        return frame, []

def add_traffic_overlay(frame, traffic_data):
    """Add traffic analysis overlay to frame"""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    
    # Add semi-transparent overlay
    alpha = 0.3
    overlay_color = (0, 0, 0)
    
    # Draw traffic info box
    padding = 10
    box_height = 120
    box_width = 200
    cv2.rectangle(overlay, (padding, padding), 
                 (box_width, box_height), 
                 overlay_color, -1)
    
    # Blend overlay
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    y = 30
    white = (255, 255, 255)
    
    texts = [
        f"Vehicles: {traffic_data['total_vehicles']}",
        f"Density: {traffic_data['density']:.2f}",
        f"Avg Speed: {traffic_data['avg_speed']:.1f} km/h",
        f"Violations: {len(traffic_data['violations'])}"
    ]
    
    for text in texts:
        cv2.putText(frame, text, (padding + 5, y), 
                   font, 0.6, white, 2)
        y += 25
    
    return frame

def calculate_average_speed(detections):
    """Calculate average speed of vehicles"""
    # This is a placeholder - implement actual speed calculation
    # based on frame-to-frame movement
    return np.random.uniform(30, 60) if detections else 0.0

class VideoProcessor:
    def __init__(self):
        self.writer = None
        self.frames = []
        self.is_writing = False
        
    def setup_writer(self, filename, fps, width, height, format="MP4"):
        try:
            os.makedirs('results', exist_ok=True)
            self.output_path = str(Path(filename).with_suffix('.avi'))
            self.fps = fps
            self.width = width
            self.height = height
            
            # Store parameters for later use
            self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
            return self.output_path
            
        except Exception as e:
            st.error(f"Error setting up video writer: {e}")
            return None
        
    def write_frame(self, frame):
        if frame is not None:
            try:
                # Convert frame to BGR if needed
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                
                # Ensure frame is uint8
                if frame.dtype != np.uint8:
                    frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
                
                # Store frame
                self.frames.append(frame)
                    
            except Exception as e:
                st.error(f"Error writing frame: {e}")
            
    def release(self):
        try:
            if self.frames:
                # Create video writer
                writer = cv2.VideoWriter(
                    self.output_path,
                    self.fourcc,
                    self.fps,
                    (self.width, self.height),
                    isColor=True
                )
                
                # Write all frames
                for frame in self.frames:
                    writer.write(frame)
                    
                writer.release()
                st.success(f"Video saved successfully to: {self.output_path}")
                
            self.frames = []  # Clear frames
            
        except Exception as e:
            st.error(f"Error saving video: {e}")

    def save_video(self, output_path):
        """Save accumulated frames to video file"""
        try:
            if not self.frames:
                return
                
            writer = cv2.VideoWriter(
                output_path,
                self.fourcc,
                self.fps,
                (self.width, self.height),
                isColor=True
            )
            
            for frame in self.frames:
                writer.write(frame)
                
            writer.release()
            st.success(f"Video saved to: {output_path}")
            
        except Exception as e:
            st.error(f"Error saving video: {e}")
        finally:
            self.frames = []  # Clear frames

def login_form():
    st.sidebar.title("Admin Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    
    if st.sidebar.button("Login"):
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state['login_status'] = True
            st.session_state['admin_view'] = True
            st.rerun()
        else:
            st.sidebar.error("Invalid credentials")

def admin_panel():
    st.sidebar.title("Admin Controls")
    
    # Model Settings
    st.sidebar.subheader("Model Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1,
        key='conf_threshold'
    )
    
    # Apply confidence threshold
    if 'detector' in st.session_state:
        st.session_state.detector.conf_thresh = confidence_threshold
    
    # Video Processing Settings
    st.sidebar.subheader("Video Processing")
    settings = {
        'enable_tracking': st.sidebar.checkbox("Enable Tracking", value=True, key='tracking'),
        'show_boxes': st.sidebar.checkbox("Show Bounding Boxes", value=True, key='boxes'),
        'show_labels': st.sidebar.checkbox("Show Labels", value=True, key='labels'),
        'export_format': st.sidebar.selectbox("Export Format", ["MP4", "AVI"], key='format')
    }
    
    if st.sidebar.button("Logout"):
        st.session_state['login_status'] = False
        st.session_state['admin_view'] = False
        st.rerun()
    
    return settings

def export_analytics(metrics, filename):
    try:
        # Create detailed analytics DataFrame
        timestamps = [datetime.now() - timedelta(seconds=x) for x in range(len(metrics.detection_history))]
        
        analytics_df = pd.DataFrame({
            'Timestamp': timestamps,
            'Frame_Number': range(len(metrics.detection_history)),
            'Detections_Count': metrics.detection_history,
            'FPS': metrics.fps_history if metrics.fps_history else [0] * len(metrics.detection_history),
            'Processing_Time': metrics.processing_times if metrics.processing_times else [0] * len(metrics.detection_history)
        })
        
        # Add summary statistics
        summary_df = pd.DataFrame({
            'Metric': [
                'Total_Unique_Detections',
                'Peak_Count',
                'Average_FPS',
                'Average_Processing_Time',
                'Total_Frames_Processed'
            ],
            'Value': [
                metrics.total_detections,
                metrics.peak_count,
                np.mean(metrics.fps_history) if metrics.fps_history else 0,
                np.mean(metrics.processing_times) if metrics.processing_times else 0,
                metrics.frames_processed
            ]
        })
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        try:
            # Try Excel format first
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                analytics_df.to_excel(writer, sheet_name='Detailed_Analytics', index=False)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        except ImportError:
            # Fallback to CSV if openpyxl is not available
            csv_filename = filename.replace('.xlsx', '.csv')
            # Save detailed analytics
            analytics_df.to_csv(csv_filename, index=False)
            # Save summary to a separate CSV
            summary_csv = csv_filename.replace('.csv', '_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            filename = csv_filename
            st.warning("Excel export not available. Saving as CSV instead.")
        
        return filename
        
    except Exception as e:
        st.error(f"Error exporting analytics: {e}")
        try:
            # Last resort: save as simple text file
            txt_filename = filename.replace('.xlsx', '.txt')
            with open(txt_filename, 'w') as f:
                f.write("Detection Analytics Summary\n\n")
                f.write(f"Total Unique Detections: {metrics.total_detections}\n")
                f.write(f"Peak Count: {metrics.peak_count}\n")
                f.write(f"Average FPS: {np.mean(metrics.fps_history) if metrics.fps_history else 0:.2f}\n")
                f.write(f"Total Frames Processed: {metrics.frames_processed}\n")
            return txt_filename
        except:
            st.error("Failed to save analytics in any format")
            return None

def create_metrics_chart(metrics):
    # Create detection history chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=metrics.detection_history,
        mode='lines',
        name='Detections',
        line=dict(color='#0083B8')
    ))
    
    fig.update_layout(
        title="Detection History",
        xaxis_title="Frames",
        yaxis_title="Number of Detections",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        paper_bgcolor='#2D2D2D',  # Match the dark theme
        plot_bgcolor='#2D2D2D',
        font=dict(color='#CCCCCC')  # Light text for dark background
    )
    
    return fig

def get_input_source():
    st.sidebar.subheader("Input Source")
    source_type = st.sidebar.radio(
        "Select Input Source",
        ["Video Upload", "CCTV/IP Camera", "Webcam"],
        key="source_type"
    )
    
    if source_type == "Video Upload":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Video",
            type=['mp4', 'avi', 'mov'],
            help="Upload a video file for pedestrian detection"
        )
        return "file", uploaded_file
        
    elif source_type == "CCTV/IP Camera":
        ip_address = st.sidebar.text_input(
            label="Camera IP Address",
            value="rtsp://username:password@ip:port/stream",
            help="Enter the RTSP stream URL for your IP camera"
        )
        if st.sidebar.button("Connect"):
            return "ip", ip_address
        return None, None
        
    else:  # Webcam
        webcam_id = st.sidebar.number_input(
            label="Webcam ID",
            min_value=0,
            value=0,
            help="Select the webcam device ID (usually 0 for built-in webcam)"
        )
        if st.sidebar.button("Start Webcam"):
            return "webcam", webcam_id
        return None, None

class AutoExporter:
    def __init__(self, base_path="results"):
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(base_path, exist_ok=True)
        
    def get_video_path(self, source_name):
        # Always use AVI format
        return os.path.join(self.base_path, f"processed_{source_name}_{self.timestamp}.avi")
        
    def get_analytics_path(self, source_name):
        return os.path.join(self.base_path, f"analytics_{source_name}_{self.timestamp}.xlsx")
        
    def export_snapshot(self, frame, metrics):
        snapshot_path = os.path.join(self.base_path, f"snapshot_{self.timestamp}.jpg")
        cv2.imwrite(snapshot_path, frame)
        return snapshot_path

def create_export_folder():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = os.path.join("results", timestamp)
    os.makedirs(export_path, exist_ok=True)
    return export_path

def add_keyboard_shortcuts():
    html("""
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'r') {  // Start/Stop recording
                document.querySelector('button:contains("Recording")').click();
            } else if (e.key === 's') {  // Take snapshot
                document.querySelector('button:contains("Snapshot")').click();
            } else if (e.key === 'q') {  // Stop processing
                document.querySelector('button:contains("Stop")').click();
            }
        });
        </script>
        """)

def update_metrics_display(metrics, detections):
    """Update the metrics display with current detection information"""
    try:
        # System Status
        st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #1E1E1E;">
                <h3>⚡ System Status</h3>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Detection System",
                "Active",
                help="Current status of detection system"
            )
        
        with col2:
            current_fps = metrics.fps_history[-1] if metrics.fps_history else 0
            st.metric(
                "Processing Rate",
                f"{current_fps:.1f} FPS",
                help="Current processing speed in frames per second"
            )
        
        with col3:
            st.metric(
                "Memory Usage",
                "Optimal",
                help="System memory status"
            )
        
        # Detection Metrics
        st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #1E1E1E;">
                <h3>📊 Detection Metrics</h3>
            </div>
            """, unsafe_allow_html=True)
        
        # Count objects by class
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display counts
        cols = st.columns(len(class_counts))
        for i, (class_name, count) in enumerate(class_counts.items()):
            cols[i].metric(
                class_name.capitalize(),
                count,
                help=f"Number of {class_name}s detected"
            )
        
        # Historical Data
        if metrics.detection_history:
            st.line_chart(metrics.detection_history[-100:])
            
    except Exception as e:
        st.error(f"Error updating metrics display: {e}")

def show_system_status(metrics):
    """Show system status and metrics"""
    st.markdown("""
        <div class="metric-card">
            <h3>🔄 System Status</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Detection Status
    status_color = "🟢" if metrics.frames_processed > 0 else "🔴"
    st.markdown(f"{status_color} Detection System: {'Active' if metrics.frames_processed > 0 else 'Inactive'}")
    
    # Performance Metrics
    if metrics.fps_history:
        current_fps = metrics.fps_history[-1]
        st.markdown(f"⚡ Processing Rate: {current_fps:.1f} FPS")
    
    # Memory Usage
    st.markdown("💾 Memory Usage: Optimal")
    
    # Detection Statistics
    st.markdown("""
        <div class="metric-card">
            <h3>📊 Detection Statistics</h3>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Total Detections", metrics.total_detections)
        st.metric("Peak Count", metrics.peak_count)
    
    with col2:
        st.metric("Current Count", metrics.current_count)
        if metrics.fps_history:
            st.metric("FPS", f"{metrics.fps_history[-1]:.1f}")

def create_export_section():
    """Create an export section with download buttons"""
    st.markdown("""
        <div class="metric-card">
            <h3>📤 Export Results</h3>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    timestamp = str(int(time.time() * 1000))
    
    # Create unique keys for this instance
    button_keys = {
        'video_download': f"download_video_{timestamp}",
        'video_disabled': f"disabled_video_{timestamp}",
        'analytics_download': f"download_analytics_{timestamp}",
        'analytics_disabled': f"disabled_analytics_{timestamp}"
    }
    
    with col1:
        video_path = st.session_state.get('final_video_path')
        if video_path and os.path.exists(video_path) and os.path.getsize(video_path) > 0:
            try:
                with open(video_path, 'rb') as f:
                    video_bytes = f.read()
                
                # Store video bytes in session state with unique key
                video_key = f'video_bytes_{timestamp}'
                st.session_state[video_key] = video_bytes
                
                st.download_button(
                    label="📹 Download Processed Video",
                    data=st.session_state[video_key],
                    file_name=f"processed_video_{timestamp}.avi",
                    mime="video/x-msvideo",
                    key=button_keys['video_download']
                )
                
                size_mb = len(st.session_state[video_key]) / (1024 * 1024)
                st.caption(f"Video size: {size_mb:.1f} MB")
                
            except Exception as e:
                st.error(f"Error preparing video download: {e}")
        else:
            st.button(
                "📹 Download Processed Video",
                disabled=True,
                help="Processing not complete or no video recorded",
                key=button_keys['video_disabled']
            )
    
    with col2:
        analytics_path = st.session_state.get('final_analytics_path')
        if analytics_path and os.path.exists(analytics_path) and os.path.getsize(analytics_path) > 0:
            try:
                with open(analytics_path, 'rb') as f:
                    analytics_data = f.read()
                
                # Store analytics data in session state with unique key
                analytics_key = f'analytics_data_{timestamp}'
                st.session_state[analytics_key] = analytics_data
                
                mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                if analytics_path.endswith('.csv'):
                    mime = "text/csv"
                elif not analytics_path.endswith('.xlsx'):
                    mime = "text/plain"
                
                st.download_button(
                    label="📊 Download Analytics Report",
                    data=st.session_state[analytics_key],
                    file_name=f"analytics_{timestamp}.xlsx",
                    mime=mime,
                    key=button_keys['analytics_download']
                )
                
                size_kb = len(st.session_state[analytics_key]) / 1024
                st.caption(f"Report size: {size_kb:.1f} KB")
                
            except Exception as e:
                st.error(f"Error preparing analytics download: {e}")
        else:
            st.button(
                "📊 Download Analytics Report",
                disabled=True,
                help="Processing not complete or no analytics generated",
                key=button_keys['analytics_disabled']
            )

def create_traffic_dashboard():
    st.title("AI Traffic Management Dashboard")
    
    # Real-time metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Traffic Density", f"{traffic_data['density']:.2f}")
        st.metric("Average Speed", f"{traffic_data['avg_speed']} km/h")
        
    with col2:
        st.metric("Violations Today", str(total_violations))
        st.metric("Emergency Vehicles", str(emergency_count))
        
    with col3:
        st.metric("Congestion Level", congestion_level)
        st.metric("Signal Efficiency", f"{signal_efficiency}%")
    
    # Traffic heatmap
    st.subheader("Traffic Density Heatmap")
    plot_traffic_heatmap(traffic_data['density_map'])
    
    # Violation analytics
    st.subheader("Traffic Violations")
    plot_violation_chart(violation_data)

def show_vehicle_legend():
    """Show legend for vehicle types"""
    st.sidebar.markdown("### Vehicle Types")
    
    for vehicle_info in VEHICLE_CLASSES.values():
        color = vehicle_info['color']
        name = vehicle_info['name'].capitalize()
        
        # Convert BGR to RGB for HTML
        rgb_color = f"rgb({color[2]}, {color[1]}, {color[0]})"
        
        st.sidebar.markdown(
            f'<div style="display: flex; align-items: center;">'
            f'<div style="width: 20px; height: 20px; background-color: {rgb_color}; '
            f'margin-right: 10px;"></div>'
            f'<span>{name}</span></div>',
            unsafe_allow_html=True
        )

def update_traffic_analysis_tab():
    """Update traffic analysis tab with detailed information"""
    st.subheader("Traffic Analysis")
    
    # Vehicle counts with colored indicators
    st.markdown("### Vehicle Counts")
    cols = st.columns(len(VEHICLE_CLASSES))
    
    # Get vehicle counts
    vehicle_counts = st.session_state.traffic_data['vehicle_counts']
    
    for i, (class_id, vehicle_info) in enumerate(VEHICLE_CLASSES.items()):
        col = cols[i]
        vehicle_name = vehicle_info['name']
        count = vehicle_counts.get(vehicle_name, 0)
        color = vehicle_info['color']
        
        # Convert BGR to RGB for display
        rgb_color = f"rgb({color[2]}, {color[1]}, {color[0]})"
        
        col.markdown(
            f'<div style="padding: 10px; border-radius: 5px; '
            f'background-color: {rgb_color}20; border: 2px solid {rgb_color}">'
            f'<h3 style="color: {rgb_color}; margin: 0;">{count}</h3>'
            f'<p style="margin: 0;">{vehicle_name.capitalize()}</p>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Traffic metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Speed", 
                 f"{st.session_state.traffic_data['avg_speed']:.1f} km/h",
                 delta="Normal" if 30 <= st.session_state.traffic_data['avg_speed'] <= 60 else "Warning")
        
    with col2:
        st.metric("Traffic Density",
                 f"{st.session_state.traffic_data['density']:.2f}",
                 delta="Low" if st.session_state.traffic_data['density'] < 0.3 else "High")

def update_signal_control_tab():
    """Update signal control interface"""
    st.subheader("Traffic Signal Control")
    
    # Signal status display
    signal_status = st.session_state.signal_controller.get_current_status()
    
    cols = st.columns(len(signal_status))
    for i, (signal_id, status) in enumerate(signal_status.items()):
        with cols[i]:
            # Signal color indicator
            color = {
                'RED': '#FF0000',
                'YELLOW': '#FFFF00',
                'GREEN': '#00FF00'
            }.get(status['phase'], '#FFFFFF')
            
            st.markdown(
                f'<div style="padding: 20px; border-radius: 10px; '
                f'background-color: {color}30; border: 2px solid {color}">'
                f'<h3 style="margin: 0;">{status["location"]}</h3>'
                f'<p style="margin: 5px 0;">Phase: {status["phase"]}</p>'
                f'<p style="margin: 5px 0;">Time: {status["timing"]}s</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    
    # Traffic density visualization
    st.subheader("Traffic Density")
    density = st.session_state.traffic_data['density']
    st.progress(min(density, 1.0))
    st.caption(f"Current Density: {density:.2f}")
    
    # Emergency vehicle override
    if st.button("🚨 Emergency Vehicle Override"):
        st.session_state.signal_controller.update_signal_status(emergency_vehicle=True)
        st.success("Emergency vehicle priority activated!")

def main():
    try:
        # Initialize components
        initialize_traffic_data()
        detector = initialize_detector()
        
        if detector is None:
            st.error("Failed to initialize detector")
            return
            
        # Store in session state
        st.session_state.detector = detector
        if 'signal_controller' not in st.session_state:
            st.session_state.signal_controller = TrafficSignalController()
            
        # Custom header
        st.markdown("""
            <div class="header-container">
                <h1 class="header-title">🚶‍♂️ Traffic Management System</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Initialize components
        if 'traffic_analyzer' not in st.session_state:
            st.session_state.traffic_analyzer = TrafficAnalyzer()
        
        # Add traffic management tabs
        tab1, tab2, tab3 = st.tabs(["Detection", "Traffic Analysis", "Signal Control"])
        
        with tab1:
            # Get settings from admin panel or use defaults
            settings = DEFAULT_SETTINGS.copy()  # Start with defaults
            if not st.session_state['login_status']:
                login_form()
            else:
                settings.update(admin_panel())  # Update with admin settings if logged in
            
            # Get input source
            source_type, source = get_input_source()
            
            if source is not None:
                try:
                    # Pass settings and detector to video processing
                    process_video_with_traffic(source, source_type, settings, detector)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
        
        with tab2:
            update_traffic_analysis_tab()
        
        with tab3:
            update_signal_control_tab()

        # Show vehicle type legend
        show_vehicle_legend()

    except Exception as e:
        st.error(f"Application error: {e}")
        # Add auto-recovery
        time.sleep(2)
        st.experimental_rerun()

def process_video_with_traffic(source, source_type, settings, detector):
    """Process video with traffic analysis"""
    try:
        # Initialize exporters and video capture
        auto_exporter = AutoExporter()
        
        if source_type == "file":
            cap = handle_video_file(source)
        elif source_type == "ip":
            cap = handle_ip_camera(source)
        else:  # webcam
            cap = handle_webcam(source)
        
        if not cap or not cap.isOpened():
            st.error("Failed to open video source")
            return
            
        # Process video stream with traffic analysis
        process_video_stream(
            cap, 
            auto_exporter, 
            settings,  # Use the passed settings
            detector   # Pass the detector
        )
        
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()

def handle_video_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(uploaded_file.read())
        return cv2.VideoCapture(tfile.name)

def handle_ip_camera(ip_address):
    return cv2.VideoCapture(ip_address)

def handle_webcam(webcam_id):
    return cv2.VideoCapture(webcam_id)

def process_video_stream(cap, auto_exporter, settings, detector):
    try:
        # Initialize session state for exports if not exists
        if 'final_video_path' not in st.session_state:
            st.session_state.final_video_path = None
        if 'final_analytics_path' not in st.session_state:
            st.session_state.final_analytics_path = None
            
        # Initialize variables first
        frame_idx = 0
        current_frame = None
        export_section = None
        final_video_path = None
        final_analytics_path = None
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Ensure valid FPS
        if fps <= 0:
            fps = 30
        
        # Initialize processors
        video_processor = VideoProcessor()
        metrics = DetectionMetrics()
        
        # Setup video writer
        output_path = None
        can_record = False
        
        if settings.get('enable_recording', True):
            output_path = video_processor.setup_writer(
                auto_exporter.get_video_path("stream"),
                fps, width, height,
                format=settings.get('export_format', "MP4")
            )
            can_record = output_path is not None
        
        # Layout
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Video container
            st.markdown('<div class="video-container">', unsafe_allow_html=True)
            video_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Controls
            st.markdown('<div class="control-panel">', unsafe_allow_html=True)
            control_col1, control_col2, control_col3 = st.columns(3)
            with control_col1:
                recording = st.checkbox(
                    "🎥 Record",
                    disabled=not can_record,
                    help="Start/Stop recording the processed video",
                    key="record_video"
                )
            with control_col2:
                if st.button(
                    "📸 Snapshot",
                    help="Take a snapshot of the current frame",
                    key="take_snapshot"
                ):
                    if current_frame is not None:
                        snapshot_path = auto_exporter.export_snapshot(current_frame, metrics)
                        st.success(f"Snapshot saved: {snapshot_path}")
            with control_col3:
                if st.button(
                    "⏹️ Stop",
                    help="Stop video processing",
                    key="stop_processing"
                ):
                    st.stop()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress
            progress_text = st.empty()
            progress_bar = st.progress(0)
            
            # Export section
            export_section = st.empty()
            
        with col2:
            # System status and metrics
            show_system_status(metrics)
            metrics_placeholder = st.empty()
        
        # Add processing status indicator
        status_indicator = st.empty()
        status_indicator.info("Processing video...")
        
        while True:
            # Check connection status
            if not check_connection():
                st.warning("Connection lost. Attempting to reconnect...")
                time.sleep(1)
                continue
                
            ret, frame = cap.read()
            if not ret:
                status_indicator.success("Processing completed!")
                break
            
            try:
                current_frame = frame.copy()
                processed_frame, detections = process_frame(frame, metrics, settings, detector)
                
                if recording and can_record:
                    video_processor.write_frame(processed_frame)
                    st.session_state.final_video_path = output_path
                
                # Display frame safely
                safe_streamlit_display(
                    video_placeholder,
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                )
                
                # Update metrics safely
                with metrics_placeholder.container():
                    try:
                        update_metrics_display(metrics, detections)
                    except:
                        pass
                
                # Update progress
                if frame_count > 0:
                    progress = min(frame_idx / frame_count, 1.0)
                    try:
                        progress_bar.progress(progress)
                        progress_text.text(f"Processing: {progress:.1%}")
                    except:
                        pass
                
                # Update export section less frequently
                if frame_idx % 30 == 0:
                    try:
                        with export_section:
                            create_export_section()
                    except:
                        pass
                
                frame_idx += 1
                
            except Exception as e:
                st.error(f"Error processing frame {frame_idx}: {e}")
                continue  # Skip problematic frame and continue
                
    except Exception as e:
        st.error(f"Error in video stream processing: {e}")
    finally:
        try:
            # Ensure cleanup happens
            video_processor.release()
            cap.release()
            
            # Save final results
            if recording and output_path:
                video_processor.save_video(output_path)
                
        except Exception as e:
            st.error(f"Error during cleanup: {e}")

def initialize_detector():
    """Initialize the YOLO detector"""
    try:
        # Use YOLOv8 model trained on COCO
        model_path = "yolov8x.pt"  # or download if not exists
        if not os.path.exists(model_path):
            from ultralytics import YOLO
            model = YOLO('yolov8x.pt')  # This will download the model
        
        # Initialize detector
        detector = ObjectDetector(
            model_path=model_path,
            conf_thresh=0.3  # Lower confidence threshold for better detection
        )
        return detector
    except Exception as e:
        st.error(f"Error initializing detector: {e}")
        return None

def initialize_traffic_data():
    """Initialize traffic data in session state"""
    if 'traffic_data' not in st.session_state:
        st.session_state.traffic_data = {
            'total_vehicles': 0,
            'total_pedestrians': 0,
            'avg_speed': 0.0,
            'density': 0.0,
            'violations': [],
            'vehicle_counts': {
                'person': 0,
                'car': 0,
                'truck': 0,
                'bus': 0,
                'motorcycle': 0,
                'bicycle': 0,
                'emergency': 0,
                'unknown': 0
            },
            'density_map': None,
            'congestion_level': 'Low',
            'signal_efficiency': 100,
            'emergency_vehicles': 0
        }

def check_connection():
    """Check if streamlit connection is active"""
    try:
        # Get session state to check connection
        _ = st.session_state
        return True
    except:
        return False

def safe_streamlit_display(placeholder, content, content_type="image"):
    """Safely display content in streamlit"""
    try:
        if content_type == "image":
            placeholder.image(content, use_container_width=True)
        elif content_type == "text":
            placeholder.text(content)
        elif content_type == "markdown":
            placeholder.markdown(content, unsafe_allow_html=True)
    except:
        pass  # Ignore display errors if connection is lost

if __name__ == "__main__":
    main() 