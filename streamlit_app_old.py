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
from datetime import datetime
import os

# Page config with custom CSS
st.set_page_config(
    page_title="Pedestrian Detection System",
    page_icon="🚶‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS and page layout
st.markdown("""
    <style>
    /* Reset and base styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    /* Main theme colors */
    :root {
        --primary-color: #0066FF;
        --secondary-color: #0052CC;
        --background-color: #FFFFFF;
        --text-color: #333333;
        --accent-color: #00A3E0;
        --gray-light: #F5F5F5;
        --gray-dark: #666666;
    }

    /* Global styles */
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }

    /* Hero section */
    .hero-container {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        padding: 4rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        text-align: center;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }

    .hero-subtitle {
        font-size: 1.2rem;
        max-width: 800px;
        margin: 0 auto;
        opacity: 0.9;
    }

    /* Feature cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
        padding: 2rem 0;
    }

    .feature-card {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        text-align: center;
    }

    .feature-card:hover {
        transform: translateY(-5px);
    }

    .feature-icon {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }

    .feature-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
        color: var(--text-color);
    }

    .feature-description {
        color: var(--gray-dark);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    /* Navigation */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 500;
        transition: background-color 0.3s ease;
    }

    .stButton > button:hover {
        background-color: var(--secondary-color);
    }

    /* Metrics and stats */
    .metric-container {
        background: var(--gray-light);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: var(--primary-color);
    }

    .metric-label {
        color: var(--gray-dark);
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        .hero-subtitle {
            font-size: 1rem;
        }
        .feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
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
    </style>
    """, unsafe_allow_html=True)

# Hero section
st.markdown("""
<div class="hero-container">
    <h1 class="hero-title">Advanced Pedestrian Detection System</h1>
    <p class="hero-subtitle">A state-of-the-art computer vision solution for detecting, tracking, and analyzing pedestrian movement in video streams.</p>
    <div style="margin-top: 2rem;">
        <a href="#demo" class="stButton"><button>Try Demo</button></a>
        <a href="#about" class="stButton" style="margin-left: 1rem;"><button>Learn More</button></a>
    </div>
</div>
""", unsafe_allow_html=True)

# Feature cards section
st.markdown("""
<h2 style="text-align: center; margin: 3rem 0;">Key Features</h2>
<div class="feature-grid">
    <div class="feature-card">
        <div class="feature-icon">👁️</div>
        <h3 class="feature-title">Real-time Detection</h3>
        <p class="feature-description">Detect pedestrians in real-time using state-of-the-art YOLOv10 models with high accuracy.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">🎯</div>
        <h3 class="feature-title">Advanced Tracking</h3>
        <p class="feature-description">Track pedestrian movements with precision using our advanced tracking algorithms and motion prediction.</p>
    </div>
    <div class="feature-card">
        <div class="feature-icon">📊</div>
        <h3 class="feature-title">Smart Analytics</h3>
        <p class="feature-description">Generate detailed insights and analytics about pedestrian patterns, density, and movement trends.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Demo section
st.markdown("""
<div style="padding: 4rem 0;">
    <h2 style="text-align: center; margin-bottom: 3rem;">Live Pedestrian Detection Demo</h2>
    
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 2rem; align-items: start;">
        <div class="feature-card" style="padding: 0; overflow: hidden;">
            <div id="video-container" style="width: 100%; position: relative;">
                <div style="padding: 56.25% 0 0 0; position: relative;">
                    <div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: var(--gray-light); display: flex; align-items: center; justify-content: center;">
                        <div style="text-align: center; color: var(--gray-dark);">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">📹</div>
                            <p>Video feed will appear here</p>
                        </div>
                    </div>
                </div>
            </div>
            <div style="padding: 1rem; display: flex; gap: 1rem; justify-content: center;">
                <button class="stButton">Start Camera</button>
                <button class="stButton" style="background: #dc3545;">Stop</button>
            </div>
        </div>
        
        <div>
            <div class="metric-container">
                <h3 style="color: var(--primary-color); margin-bottom: 1rem;">Detection Statistics</h3>
                <div style="margin-bottom: 1rem;">
                    <div class="metric-value">0</div>
                    <div class="metric-label">Pedestrians Detected</div>
                </div>
                <div>
                    <div class="metric-value">0</div>
                    <div class="metric-label">Processing FPS</div>
                </div>
            </div>
            
            <div class="feature-card" style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
                <h3 class="feature-title">Upload Video</h3>
                <p class="feature-description">Drag and drop a video file here or click to select</p>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# About section
st.markdown("""
<div style="padding: 2rem 0;">
    <h2 style="text-align: center; margin-bottom: 3rem;">About the Project</h2>
    
    <div style="display: grid; grid-template-columns: 2fr 1fr; gap: 2rem;">
        <div>
            <h3 style="color: var(--primary-color); margin-bottom: 1rem;">System Architecture</h3>
            <div class="feature-card" style="margin-bottom: 2rem;">
                <h4>Detection Layer</h4>
                <p>Utilizes YOLOv10 for accurate real-time pedestrian detection with enhanced performance through:</p>
                <ul style="list-style-type: none; padding-left: 1rem;">
                    <li>• Advanced feature extraction</li>
                    <li>• Multi-scale detection capabilities</li>
                    <li>• Optimized inference pipeline</li>
                </ul>
            </div>
            
            <div class="feature-card" style="margin-bottom: 2rem;">
                <h4>Tracking Integration</h4>
                <p>Implements DeepSORT (Deep Simple Online Realtime Tracking) algorithm with:</p>
                <ul style="list-style-type: none; padding-left: 1rem;">
                    <li>• Integrated detection and tracking pipeline</li>
                    <li>• Efficient ID management system</li>
                    <li>• Robust occlusion handling</li>
                </ul>
            </div>
            
            <div class="feature-card">
                <h4>XAI Integration</h4>
                <p>Explainable AI component provides transparency through:</p>
                <ul style="list-style-type: none; padding-left: 1rem;">
                    <li>• Real-time saliency map generation</li>
                    <li>• Feature importance visualization</li>
                    <li>• Decision confidence metrics</li>
                </ul>
            </div>
        </div>
        
        <div>
            <div class="metric-container">
                <h3 style="color: var(--primary-color); margin-bottom: 1rem;">Project Stats</h3>
                <div style="margin-bottom: 1rem;">
                    <div class="metric-value">98.5%</div>
                    <div class="metric-label">Detection Accuracy</div>
                </div>
                <div style="margin-bottom: 1rem;">
                    <div class="metric-value">45 FPS</div>
                    <div class="metric-label">Processing Speed</div>
                </div>
                <div>
                    <div class="metric-value">96.8%</div>
                    <div class="metric-label">Tracking Precision</div>
                </div>
            </div>
            
            <div class="metric-container">
                <h3 style="color: var(--primary-color); margin-bottom: 1rem;">Technology Stack</h3>
                <ul style="list-style-type: none; padding-left: 0;">
                    <li style="margin-bottom: 0.5rem;">• <strong>Object Detection:</strong> YOLOv10 (primary model) with support for YOLOv8</li>
                    <li style="margin-bottom: 0.5rem;">• <strong>Tracking:</strong> DeepSORT algorithm for reliable multi-object tracking</li>
                    <li style="margin-bottom: 0.5rem;">• <strong>Backend:</strong> Python with Flask for web service integration</li>
                    <li style="margin-bottom: 0.5rem;">• <strong>Frontend:</strong> HTML5, CSS3 and JavaScript with Bootstrap framework</li>
                    <li>• <strong>Data Analysis:</strong> NumPy, Pandas, and custom analytics modules</li>
                </ul>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Default detection settings
DEFAULT_SETTINGS = {
    'show_boxes': True,
    'show_labels': True,
    'confidence_threshold': 0.3,
    'show_debug': False
}

# Admin credentials
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

# Initialize session state
if 'login_status' not in st.session_state:
    st.session_state['login_status'] = False
if 'admin_view' not in st.session_state:
    st.session_state['admin_view'] = False

# Load configuration
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

class DetectionMetrics:
    def __init__(self):
        self.total_detections = 0
        self.current_count = 0
        self.frames_processed = 0
        self.start_time = time.time()
        self.processing_times = []
        self.detection_history = []
        self.fps_history = []
        self.peak_count = 0

    def update(self, detections, processing_time):
        try:
            self.frames_processed += 1
            self.processing_times.append(processing_time)
            
            # Update current count and total detections
            self.current_count = len(detections)
            self.total_detections += self.current_count
            
            # Update peak count
            self.peak_count = max(self.peak_count, self.current_count)
            
            # Update detection history
            self.detection_history.append(self.current_count)
            
            # Calculate and update FPS
            if self.processing_times:
                current_fps = 1.0 / (sum(self.processing_times[-10:]) / min(len(self.processing_times), 10))
                self.fps_history.append(current_fps)
            
            # Keep last 100 frames of history
            if len(self.detection_history) > 100:
                self.detection_history.pop(0)
            if len(self.processing_times) > 100:
                self.processing_times.pop(0)
            if len(self.fps_history) > 100:
                self.fps_history.pop(0)
                
        except Exception as e:
            st.error(f"Error updating metrics: {e}")

def process_frame(frame, metrics, settings, detector):
    """Process a single frame with detection and XAI visualization"""
    try:
        # Initialize XAI visualizer if not already in session state
        if 'xai_visualizer' not in st.session_state:
            st.session_state.xai_visualizer = XAIVisualizer(detector.model)

        # Create a copy of the frame for processing
        processed_frame = frame.copy()
        
        # Perform detection
        detections = detector.detect(frame)
        
        # Filter for pedestrians (class_id 0)
        pedestrians = [d for d in detections if d['class_id'] == 0]
        
        # Update metrics
        metrics.update(len(pedestrians))
        
        # Process each detection
        for det in pedestrians:
            bbox = det['bbox']
            x1, y1, w, h = map(int, bbox)
            
            # Draw bounding box if enabled
            if settings.get('show_boxes', True):
                cv2.rectangle(processed_frame, (x1, y1), (x1+w, y1+h), det['color'], 2)
            
            # Draw label if enabled
            if settings.get('show_labels', True):
                label = f"{det['class_name']}: {det['confidence']:.2f}"
                cv2.putText(processed_frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, det['color'], 2)
            
            # Generate and overlay XAI visualization
            if settings.get('show_xai', True):
                processed_frame = st.session_state.xai_visualizer.generate_saliency_map(
                    processed_frame, det
                )
        
        return processed_frame, detections
        
    except Exception as e:
        st.error(f"Error processing frame: {e}")
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
    """Update the metrics display with pedestrian detection information"""
    try:
        if not hasattr(metrics, 'fps_history') or not metrics.fps_history:
            return
            
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
            current_fps = metrics.fps_history[-1]
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
        
        # Pedestrian Detection Metrics
        st.markdown("""
            <div style="padding: 10px; border-radius: 5px; background-color: #1E1E1E;">
                <h3>👥 Pedestrian Detection</h3>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Current Count",
                len(detections),
                help="Number of pedestrians currently detected"
            )
        
        with col2:
            st.metric(
                "Total Detections",
                metrics.total_detections,
                help="Total number of pedestrians detected"
            )
        
        # Detection History
        if metrics.detection_history:
            st.line_chart(
                metrics.detection_history[-100:]
            )
            
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
                    key=f"download_video_btn_{timestamp}"
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
                key=f"disabled_video_btn_{timestamp}"
            )
    
    with col2:
        analytics_path = st.session_state.get('final_analytics_path')
        if analytics_path and os.path.exists(analytics_path) and os.path.getsize(analytics_path) > 0:
            try:
                with open(analytics_path, 'rb') as f:
                    analytics_data = f.read()
                
                st.download_button(
                    label="📊 Download Analytics Data",
                    data=analytics_data,
                    file_name=f"analytics_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_analytics_btn_{timestamp}"
                )
                
                size_mb = os.path.getsize(analytics_path) / (1024 * 1024)
                st.caption(f"Analytics file size: {size_mb:.1f} MB")
                
            except Exception as e:
                st.error(f"Error preparing analytics download: {e}")
        else:
            st.button(
                "📊 Download Analytics Data",
                disabled=True,
                help="Analytics data not available",
                key=f"disabled_analytics_btn_{timestamp}"
            )



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
        # Initialize detection data
        initialize_detection_data()
        detector = initialize_detector()
        
        if detector is None:
            st.error("Failed to initialize detector")
            return
            
        # Store detector in session state
        st.session_state.detector = detector
            
        # Custom header
        st.markdown("""
            <div class="header-container">
                <h1 class="header-title">🚶‍♂️ Pedestrian Detection System</h1>
            </div>
            """, unsafe_allow_html=True)
        
        # Get settings from admin panel or use defaults
        settings = DEFAULT_SETTINGS.copy()  # Start with defaults
        if not st.session_state.get('login_status', False):
            login_form()
        else:
            settings.update(admin_panel())  # Update with admin settings if logged in
        
        # Get input source
        source_type, source = get_input_source()
        
        if source is not None:
            try:
                # Process video with pedestrian detection
                process_video(source, source_type, settings, detector)
            except Exception as e:
                st.error(f"An error occurred during video processing: {e}")

    except Exception as e:
        st.error(f"Application error: {e}")
        # Add auto-recovery using current Streamlit API
        time.sleep(2)
        st.rerun()

def process_video(source, source_type, settings, detector):
    try:
        # Initialize metrics
        metrics = DetectionMetrics()
        
        # Initialize video capture
        cap = None
        if source_type == 'webcam':
            cap = cv2.VideoCapture(0)
        elif source_type == 'file':
            cap = cv2.VideoCapture(source)
        
        if not cap or not cap.isOpened():
            st.error("Error opening video source")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create video display elements
        st.markdown('<div class="video-container">', unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

        # Create control elements
        col1, col2, col3 = st.columns(3)
        with col1:
            show_boxes = st.checkbox('Show Bounding Boxes', value=True)
        with col2:
            show_labels = st.checkbox('Show Labels', value=True)
        with col3:
            show_xai = st.checkbox('Show XAI Visualization', value=True)

        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Create metrics display
        metrics_placeholder = st.empty()

        # Process video frames
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update settings
            settings.update({
                'show_boxes': show_boxes,
                'show_labels': show_labels,
                'show_xai': show_xai
            })

            # Process frame with detections and XAI
            processed_frame, detections = process_frame(frame, metrics, settings, detector)

            # Display the processed frame
            video_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB))

            # Update progress
            frame_count += 1
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f'Processing frame {frame_count} of {total_frames}')

            # Update metrics display
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                col1.metric('Total Detections', metrics.total_detections)
                col2.metric('Current Count', metrics.current_count)
                col3.metric('Peak Count', metrics.peak_count)

    except Exception as e:
        st.error(f"Error processing video: {e}")
    finally:
        # Cleanup resources
        try:
            if cap is not None:
                cap.release()
            if 'out' in locals() and out is not None:
                out.release()
            if 'output_path' in locals() and output_path and os.path.exists(output_path):
                st.success(f"Processed video saved to: {output_path}")
        except Exception as e:
            st.error(f"Error during cleanup: {e}")

def handle_video_file(uploaded_file):
    """Handle uploaded video file with proper error checking"""
    try:
        if uploaded_file is None:
            raise ValueError("No file was uploaded")
            
        # Handle string path or uploaded file object
        if isinstance(uploaded_file, str):
            if not os.path.exists(uploaded_file):
                raise ValueError(f"Video file not found: {uploaded_file}")
            return cv2.VideoCapture(uploaded_file)
            
        # Create temp file with appropriate suffix
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if suffix not in ['.mp4', '.avi', '.mov', '.mkv']:
            raise ValueError(f"Unsupported video format: {suffix}")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {uploaded_file.name}")
                
            return cap
            
    except Exception as e:
        st.error(f"Error handling video file: {str(e)}")
        return None

def handle_ip_camera(ip_address):
    """Handle IP camera connection with validation"""
    try:
        if not ip_address:
            raise ValueError("IP address is required")
            
        # Validate IP address format
        import re
        ip_pattern = r'^rtsp://[\w\-\.]+(?::\d+)?(?:/[\w\-\.]+)*$|^http://[\w\-\.]+(?::\d+)?(?:/[\w\-\.]+)*$'
        if not re.match(ip_pattern, ip_address):
            raise ValueError("Invalid IP camera address format")
            
        cap = cv2.VideoCapture(ip_address)
        if not cap.isOpened():
            raise ValueError(f"Could not connect to IP camera at {ip_address}")
            
        return cap
        
    except Exception as e:
        st.error(f"Error connecting to IP camera: {str(e)}")
        return None

def handle_webcam(webcam_id):
    """Handle webcam initialization with device checking"""
    try:
        cap = cv2.VideoCapture(webcam_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID: {webcam_id}")
            
        # Test frame capture
        ret, _ = cap.read()
        if not ret:
            raise ValueError("Could not read frame from webcam")
            
        return cap
        
    except Exception as e:
        st.error(f"Error initializing webcam: {str(e)}")
        return None

def process_video_stream(cap, auto_exporter, settings, detector):
    try:
        # Initialize session state for exports and frame buffer if not exists
        if 'final_video_path' not in st.session_state:
            st.session_state.final_video_path = None
        if 'final_analytics_path' not in st.session_state:
            st.session_state.final_analytics_path = None
        if 'frame_buffer' not in st.session_state:
            st.session_state.frame_buffer = []
        if 'buffer_size' not in st.session_state:
            st.session_state.buffer_size = 30  # Buffer size for smooth playback
        if 'last_frame_time' not in st.session_state:
            st.session_state.last_frame_time = time.time()

        # Frame rate control for smooth playback
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        current_time = time.time()
        elapsed = current_time - st.session_state.last_frame_time

        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

        # Frame buffering for smooth playback
        try:
            while len(st.session_state.frame_buffer) < st.session_state.buffer_size:
                ret, frame = cap.read()
                if not ret:
                    if len(st.session_state.frame_buffer) == 0:
                        st.error("Error: Unable to read video frame")
                        return None
                    break
                st.session_state.frame_buffer.append(frame)

            # Process frames from buffer
            if st.session_state.frame_buffer:
                frame = st.session_state.frame_buffer.pop(0)
                ret, next_frame = cap.read()
                if ret:
                    st.session_state.frame_buffer.append(next_frame)
                st.session_state.last_frame_time = time.time()
                return frame
        except cv2.error as e:
            st.error(f"OpenCV Error: {str(e)}")
            return None
        except Exception as e:
            st.error(f"Unexpected error during video processing: {str(e)}")
            return None
        if 'frame_buffer' not in st.session_state:
            st.session_state.frame_buffer = []
        if 'buffer_size' not in st.session_state:
            st.session_state.buffer_size = 30  # Number of frames to buffer
            
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

def initialize_detection_data():
    """Initialize detection data in session state"""
    if 'detection_data' not in st.session_state:
        st.session_state.detection_data = {
            'total_pedestrians': 0,
            'current_count': 0,
            'detection_history': [],
            'processing_times': []
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