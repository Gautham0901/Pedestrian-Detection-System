from flask import Flask, render_template, jsonify, Response, request
from modules.detector import ObjectDetector
import yaml
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import logging
from datetime import datetime
import time
from time import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Load configuration
with open('configs/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Initialize detector with configuration
model_path = config.get('model_path', 'models/yolo11s.pt')
conf_thresh = config.get('detection', {}).get('confidence_threshold', 0.5)
detector = ObjectDetector(model_path=model_path, conf_thresh=conf_thresh)

# Global video capture object
video_capture = None
is_playing = False
current_frame = 0

# Add these variables to track metrics
class DetectionMetrics:
    def __init__(self):
        self.total_detections = 0
        self.frames_processed = 0
        self.start_time = time()
        self.last_frame_time = time()
        self.processing_times = []

metrics = DetectionMetrics()

class VideoProcessor:
    def __init__(self, detector):
        self.detector = detector
        self.frame_count = 0
        self.anomaly_count = 0
        self.start_time = datetime.now()
        
    def process_frame(self, frame):
        """Process frame with detection and visualization"""
        if frame is None:
            return None, []
            
        try:
            # Perform detection
            detections = self.detector.detect(frame)
            
            # Update metrics
            metrics.total_detections += len(detections)
            metrics.frames_processed += 1
            processing_time = time() - frame_start_time
            metrics.processing_times.append(processing_time)
            if len(metrics.processing_times) > 30:  # Keep last 30 frames for average
                metrics.processing_times.pop(0)
            
            # Draw detections
            for det in detections:
                bbox, conf, class_id = det
                x1, y1, w, h = bbox
                
                # Convert tensor values to float/int
                conf_value = float(conf)  # Convert confidence tensor to float
                class_id_value = int(class_id)  # Convert class_id tensor to int
                
                # Draw bounding box
                cv2.rectangle(frame, 
                            (int(x1), int(y1)), 
                            (int(x1+w), int(y1+h)), 
                            (0, 255, 0), 2)
                
                # Draw label with proper formatting
                label = f"{self.detector.get_class_names()[class_id_value]}: {conf_value:.2f}"
                cv2.putText(frame, label, 
                           (int(x1), int(y1-10)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
            
            return frame, detections
            
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return frame, []

def get_video_stream():
    global video_capture, is_playing, current_frame
    processor = VideoProcessor(detector)
    
    while True:
        try:
            if video_capture is None or not video_capture.isOpened():
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No video loaded", (180, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            else:
                if not is_playing:
                    # If paused, keep showing the current frame
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                
                ret, frame = video_capture.read()
                if not ret:
                    # Reset video to beginning
                    current_frame = 0
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                current_frame = int(video_capture.get(cv2.CAP_PROP_POS_FRAMES))
                
                if is_playing:
                    # Process frame only if playing
                    frame, _ = processor.process_frame(frame)
                    if frame is None:
                        continue

            # Convert frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                   
        except Exception as e:
            logger.error(f"Error in video stream: {e}")
            time.sleep(0.1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(get_video_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload', methods=['POST'])
def upload_file():
    global video_capture
    
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Close existing video capture
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Open new video capture
        video_capture = cv2.VideoCapture(filepath)
        
        if not video_capture.isOpened():
            raise Exception("Failed to open video file")
        
        # Get video properties
        width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video loaded: {filename} ({width}x{height} @ {fps}fps)")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'info': {
                'width': width,
                'height': height,
                'fps': fps,
                'frames': frame_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading video: {e}")
        if video_capture is not None:
            video_capture.release()
            video_capture = None
        return jsonify({'error': str(e)}), 500

@app.route('/api/detections')
def get_detections():
    global video_capture
    if video_capture is not None and video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            detections = detector.detect(frame)
            # Reset frame position
            current_pos = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
            
            return jsonify([{
                'class_name': detector.get_class_names()[det[2]],
                'confidence': float(det[1]),
                'bbox': det[0]
            } for det in detections])
    return jsonify([])

@app.route('/api/stats')
def get_stats():
    global metrics
    current_time = time()
    elapsed_time = current_time - metrics.start_time
    
    # Calculate FPS
    if metrics.processing_times:
        current_fps = 1.0 / (sum(metrics.processing_times) / len(metrics.processing_times))
    else:
        current_fps = 0
    
    # Calculate detection rate
    detection_rate = metrics.total_detections / max(metrics.frames_processed, 1)
    
    return jsonify({
        'total_detections': metrics.total_detections,
        'detection_rate': detection_rate,
        'processing_fps': current_fps,
        'elapsed_time': elapsed_time,
        'average_processing_time': sum(metrics.processing_times) / len(metrics.processing_times) if metrics.processing_times else 0
    })

@app.route('/api/controls/<action>', methods=['POST'])
def video_controls(action):
    global video_capture, is_playing, current_frame
    
    try:
        if video_capture is None:
            return jsonify({'error': 'No video loaded'}), 400
            
        if action == 'play':
            is_playing = True
            return jsonify({'status': 'playing'})
            
        elif action == 'pause':
            is_playing = False
            return jsonify({'status': 'paused'})
            
        elif action == 'stop':
            is_playing = False
            current_frame = 0
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return jsonify({'status': 'stopped'})
            
        elif action == 'seek':
            position = request.json.get('position', 0)  # Position in percentage
            if not 0 <= position <= 100:
                return jsonify({'error': 'Invalid position'}), 400
                
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_pos = int((position / 100) * total_frames)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            current_frame = frame_pos
            return jsonify({'status': 'seeked', 'frame': frame_pos})
            
    except Exception as e:
        logger.error(f"Error in video controls: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if video_capture is not None:
            video_capture.release() 