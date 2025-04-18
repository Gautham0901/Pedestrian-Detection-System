from flask import render_template, request, jsonify, Response, redirect, url_for
import os
import sys
import cv2
import numpy as np
import time
import json
from datetime import datetime
import threading

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.detector import ObjectDetector
from modules.tracker import TrackerModule
from modules.analytics import Analytics
from modules.xai import XAIVisualizer
from flask import Flask
from routes import routes
from socket_app import create_socket_app

app = Flask(__name__)
app.register_blueprint(routes)

# Initialize SocketIO
socketio = create_socket_app(app)

# Global variables
detector = None
tracker = None
xai_visualizer = None
video_path = None
output_path = None
processing_thread = None
is_processing = False
comparison_data = None
processing_stats = {"pedestrian_count": 0, "processing_time": 0, "processing_fps": 0, "progress": 0}
analytics_data = None

# Initialize detector and tracker
def initialize_models():
    global detector, tracker, xai_visualizer, analytics_data
    try:
        detector = ObjectDetector(model_path="yolov10x.pt", conf_thresh=0.3)
        tracker = TrackerModule(max_age=20, n_init=3)
        xai_visualizer = XAIVisualizer(detector.model)
    except Exception as e:
        app.logger.error(f"Error initializing models: {str(e)}")
        raise
    
    # Initialize analytics data
    analytics_data = {
        "pedestrian_flow": {
            "weekday": [120, 190, 210, 230, 280, 290, 220, 210, 240, 290, 270, 180],
            "weekend": [80, 120, 150, 180, 210, 200, 190, 195, 210, 200, 180, 150],
            "labels": ['8:00', '9:00', '10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00', '17:00', '18:00', '19:00']
        },
        "dwell_time": {
            "zones": ['Shopping', 'Transit', 'Recreation', 'Dining', 'Services'],
            "times": [12.5, 3.2, 18.7, 22.3, 8.1]
        },
        "anomalies": {
            "normal": [220, 240, 230, 250, 260, 190, 180],
            "threshold": [300, 300, 300, 300, 300, 300, 300],
            "detected": [None, 320, None, None, 350, None, None],
            "days": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        },
        "demographics": {
            "categories": ['Adults', 'Children', 'Elderly', 'Groups'],
            "distribution": [68, 12, 15, 5]
        },
        "zones": [
            {"name": "Zone A (Entrance)", "density": "High", "trend": 12, "trend_direction": "up"},
            {"name": "Zone B (Central)", "density": "Medium", "trend": 5, "trend_direction": "down"},
            {"name": "Zone C (Exit)", "density": "High", "trend": 8, "trend_direction": "up"},
            {"name": "Zone D (Retail)", "density": "Very High", "trend": 15, "trend_direction": "up"}
        ]
    }
    
    # Load comparison data
    load_comparison_data()

def load_comparison_data():
    global comparison_data
    comparison_data = {
        "models": [
            {"name": "YOLOv10", "precision": 0.94, "recall": 0.91, "f1": 0.92, "fps": 22},
            {"name": "YOLOv8", "precision": 0.92, "recall": 0.89, "f1": 0.90, "fps": 25},
            {"name": "SSD", "precision": 0.85, "recall": 0.82, "f1": 0.83, "fps": 30},
            {"name": "Faster R-CNN", "precision": 0.91, "recall": 0.88, "f1": 0.89, "fps": 15}
        ],
        "trackers": [
            {"name": "DeepSORT", "accuracy": 0.88, "id_switches": 12, "fps": 28},
            {"name": "SORT", "accuracy": 0.82, "id_switches": 25, "fps": 35},
            {"name": "ByteTrack", "accuracy": 0.90, "id_switches": 8, "fps": 26}
        ]
    }

# Process video in a separate thread
def process_video_thread(input_path, output_path):
    global is_processing, processing_stats, detector, tracker
    
    try:
        # Ensure detector and tracker are initialized
        if detector is None or tracker is None:
            initialize_models()
            
        # Initialize processing stats
        processing_stats = {
            "pedestrian_count": 0,
            "processing_time": 0,
            "processing_fps": 0,
            "progress": 0
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, 
            fourcc, 
            fps, 
            (width, height)
        )
        
        if not writer.isOpened():
            raise Exception(f"Could not create output video file: {output_path}")
        
        frame_count = 0
        start_time = time.time()
        pedestrian_count = 0
        
        while is_processing:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Update progress
            if total_frames > 0:
                progress = min(round((frame_count / total_frames) * 100), 99)
                processing_stats["progress"] = progress
                
            # Detect only pedestrians
            detections = detector.detect(frame)
            pedestrians = [d for d in detections if d['class_id'] == 0]
            pedestrian_count += len(pedestrians)
            
            # Process detections and draw visualizations
            for det in pedestrians:
                bbox = det['bbox']
                x1, y1, w, h = bbox
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
                
                # Draw label
                label = f"Person: {det['confidence']:.2f}"
                cv2.putText(frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                
                # Generate and overlay XAI visualization
                frame = xai_visualizer.generate_saliency_map(frame.copy(), det)
            
            # Draw detections and tracks
            for det in detections:
                if det['class_id'] == 0:  # Only show pedestrians
                    x, y, w, h = det['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    label = f"Person: {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                track_id = track.track_id
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {track_id}", (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Write frame to output
            writer.write(frame)
            frame_count += 1
            
        # Calculate processing stats
        elapsed_time = time.time() - start_time
        processing_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Update final processing stats
        processing_stats = {
            "pedestrian_count": pedestrian_count,
            "processing_time": round(elapsed_time, 1),
            "processing_fps": round(processing_fps, 1),
            "progress": 100
        }
        
        print(f"Video processing completed. Processed {frame_count} frames at {processing_fps:.2f} FPS")
        
    except Exception as e:
        print(f"Error processing video: {e}")
        processing_stats["error"] = str(e)
    finally:
        is_processing = False
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'writer' in locals() and writer is not None:
            writer.release()

# Route handlers
@app.route('/')
def index():
    try:
        if detector is None or tracker is None:
            initialize_models()
        return render_template('index.html')
    except Exception as e:
        app.logger.error(f"Error in index route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/about')
def about():
    try:
        if detector is None or tracker is None:
            initialize_models()
        return render_template('about.html')
    except Exception as e:
        app.logger.error(f"Error in about route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/showcase')
def showcase():
    try:
        if detector is None or tracker is None:
            initialize_models()
        return render_template('showcase.html', comparison_data=comparison_data)
    except Exception as e:
        app.logger.error(f"Error in showcase route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/demo')
def demo():
    try:
        if detector is None or tracker is None:
            initialize_models()
        return render_template('demo.html')
    except Exception as e:
        app.logger.error(f"Error in demo route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analysis')
def analysis():
    try:
        if detector is None or tracker is None or analytics_data is None:
            initialize_models()
        return render_template('analysis.html', analytics_data=analytics_data, comparison_data=comparison_data)
    except Exception as e:
        app.logger.error(f"Error in analysis route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/documentation')
def documentation():
    try:
        if detector is None or tracker is None:
            initialize_models()
        return render_template('documentation.html')
    except Exception as e:
        app.logger.error(f"Error in documentation route: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

# API endpoints
@app.route('/process_video', methods=['POST'])
def api_process_video():
    global video_path, output_path, processing_thread, is_processing, processing_stats
    
    if is_processing:
        return jsonify({"status": "error", "message": "Already processing a video"})
    
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"})
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({"status": "error", "message": "No video file selected"})
        
    # Validate video file format
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        return jsonify({"status": "error", "message": "Invalid video format. Please upload MP4, AVI, MOV, or MKV files."})
    
    try:
        # Create necessary directories if they don't exist
        videos_dir = os.path.join(app.root_path, 'static', 'videos')
        os.makedirs(videos_dir, exist_ok=True)
        
        # Save the uploaded video with timestamp to avoid conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f'input_{timestamp}.mp4'
        video_path = os.path.join(videos_dir, video_filename)
        
        # Verify directory exists and is writable
        if not os.path.exists(videos_dir) or not os.access(videos_dir, os.W_OK):
            raise Exception(f"Videos directory {videos_dir} does not exist or is not writable")
            
        # Save the file and verify it was saved successfully
        video_file.save(video_path)
        if not os.path.exists(video_path) or not os.path.isfile(video_path):
            raise Exception("Failed to save video file")
            
        # Verify file is readable
        try:
            with open(video_path, 'rb') as f:
                f.read(1024)  # Try reading first 1KB
        except IOError:
            raise Exception("Saved file is not readable")
        
        # Set output path
        output_filename = f'output_{timestamp}.mp4'
        output_path = os.path.join(videos_dir, output_filename)
        
        # Reset processing stats
        processing_stats = {
            "pedestrian_count": 0,
            "processing_time": 0,
            "processing_fps": 0,
            "progress": 0
        }
        
        # Start processing in a separate thread
        is_processing = True
        processing_thread = threading.Thread(
            target=process_video_thread,
            args=(video_path, output_path)
        )
        processing_thread.daemon = True
        processing_thread.start()
        
        return jsonify({
            "status": "success", 
            "message": "Video processing started",
            "output_video": f'/static/videos/{output_filename}'
        })
    except Exception as e:
        is_processing = False
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Error processing video: {str(e)}"
        })
    finally:
        if video_path and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception as e:
                app.logger.error(f"Error removing temporary video file: {e}")

@app.route('/api/processing-status')
def api_processing_status():
    global is_processing, processing_stats
    return jsonify({
        "is_processing": is_processing,
        "stats": processing_stats
    })

@app.route('/api/analytics-data')
def api_analytics_data():
    global analytics_data
    if analytics_data is None:
        initialize_models()
    return jsonify(analytics_data)

# Add new route for video streaming
@app.route('/static/videos/<path:filename>')
def serve_video(filename):
    video_path = os.path.join(app.root_path, 'static', 'videos', filename)
    
    # Verify file exists
    if not os.path.exists(video_path):
        return "Video not found", 404
        
    def generate():
        with open(video_path, 'rb') as video:
            data = video.read(1024*1024)  # Read 1MB chunks
            while data:
                yield data
                data = video.read(1024*1024)
    
    response = Response(generate(), mimetype='video/mp4')
    response.headers['Content-Type'] = 'video/mp4'
    response.headers['Accept-Ranges'] = 'bytes'
    return response

@app.after_request
def add_header(response):
    response.headers['Accept-Ranges'] = 'bytes'
    if 'video' in response.mimetype:
        response.headers['Content-Type'] = 'video/mp4'
    return response

if __name__ == '__main__':
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)