import os
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file, current_app
from flask_socketio import SocketIO
from datetime import datetime
from pathlib import Path
import pandas as pd

from modules.detector import ObjectDetector
from modules.xai import XAIVisualizer
from modules.analytics import Analytics
from modules.report_generator import ReportGenerator
from modules.integrated_detector import IntegratedDetectorTracker

from flask import Blueprint, render_template

routes = Blueprint('routes', __name__)
socketio = SocketIO()

# Initialize models and create required directories
detector = ObjectDetector(model_path='yolov10x.pt', conf_thresh=0.3)
xai_visualizer = XAIVisualizer(detector.model)
analytics = Analytics(frame_size=(720, 1280))  # Standard HD resolution

# Create required directories
# Create required directories with absolute paths
base_dir = Path(__file__).parent.resolve()
upload_dir = (base_dir / 'static' / 'videos').resolve()
result_dir = (base_dir.parent / 'results').resolve()
for directory in [upload_dir, result_dir]:
    directory.mkdir(parents=True, exist_ok=True)

@routes.route('/api/upload', methods=['POST'])
def upload_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'error': 'No selected file'}), 400
            
        # Generate unique timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Use the already defined absolute upload directory
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate file type
        if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov')):
            return jsonify({'error': 'Invalid video format. Please upload MP4, AVI, or MOV files.'}), 400
            
        # Save uploaded file with secure filename
        video_path = upload_dir / f'input_{timestamp}.mp4'
        try:
            # Ensure upload directory exists
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the file
            video_file.save(str(video_path))
            
            # Verify file was saved successfully
            if not video_path.exists() or not video_path.is_file():
                return jsonify({'error': 'Failed to save video file'}), 500
                
            # Verify file is readable
            try:
                with open(video_path, 'rb') as f:
                    f.read(1024)  # Try reading first 1KB
            except IOError:
                return jsonify({'error': 'Saved file is not readable'}), 500
        except Exception as e:
            return jsonify({'error': f'Error saving video file: {str(e)}'}), 500
        
        # Process video synchronously for now
        process_video(str(video_path), timestamp)
        
        return jsonify({
            'message': 'Video uploaded successfully',
            'job_id': timestamp
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Global progress tracking
processing_progress = {}

@routes.route('/api/status/<job_id>')
def get_status(job_id):
    try:
        report_path = result_dir / f'report_{job_id}.pdf'
        if report_path.exists():
            # Clear progress data when complete
            if job_id in processing_progress:
                del processing_progress[job_id]
            return jsonify({
                'status': 'completed',
                'report_url': f'/api/download/{job_id}'
            })
        # Return progress if available
        progress = processing_progress.get(job_id, 0)
        return jsonify({
            'status': 'processing',
            'progress': progress
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/api/download/<job_id>')
def download_report(job_id):
    try:
        report_path = result_dir / f'report_{job_id}.pdf'
        if not report_path.exists():
            return jsonify({'error': 'Report not found'}), 404
        return send_file(
            str(report_path),
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'pedestrian_analysis_report_{job_id}.pdf'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video(video_path, job_id):
    try:
        global processing_progress
        processing_progress[job_id] = 0
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            del processing_progress[job_id]  # Clear progress on error
            raise ValueError('Could not open video file')
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            del processing_progress[job_id]  # Clear progress if invalid frame count
            raise ValueError('Invalid video file: no frames detected')
            
        # Initialize results storage
        detections_data = []
        frame_count = 0
        total_processing_time = 0
        xai_frames = []  # Store frames with XAI visualizations
        
        # Get video properties for analytics
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        analytics.update_frame_size((height, width))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            start_time = datetime.now()
            
            # Detect and track pedestrians using integrated detector
            results = detector.detect_and_track(frame)
            
            # Process each detection
            frame_with_xai = frame.copy()
            for det in results:
                if det['class_name'] == 'person':
                    # Generate XAI visualization
                    frame_with_xai = xai_visualizer.generate_saliency_map(frame_with_xai, det)
                    
                    # Store detection data
                    detection_info = {
                        'frame': frame_count,
                        'confidence': det['confidence'],
                        'bbox': det['bbox'],
                        'track_id': det.get('track_id', None)
                    }
                    detections_data.append(detection_info)
                    
                    # Draw detection visualization
                    x1, y1, w, h = det['bbox']
                    cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                    label = f"Person {det.get('track_id', '')}: {det['confidence']:.2f}"
                    cv2.putText(frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Store XAI frames at regular intervals or for significant detections
            if len(results) > 0 and (frame_count % 30 == 0 or len(results) >= 3):
                xai_frames.append(frame_with_xai)
                
            frame_count += 1
            total_processing_time += (datetime.now() - start_time).total_seconds()
            
            # Update progress
            if total_frames > 0:
                processing_progress[job_id] = int((frame_count / total_frames) * 100)
            
        # Generate report
        df = pd.DataFrame(detections_data)
        
        # Calculate statistics
        stats = {
            'total_frames': frame_count,
            'total_detections': len(detections_data),
            'avg_confidence': df['confidence'].mean() if not df.empty else 0,
            'avg_processing_time': total_processing_time / frame_count,
            'fps': frame_count / total_processing_time
        }
        
        # Generate PDF report with XAI visualizations
        report_generator = ReportGenerator(result_dir)
        report_path = report_generator.generate_report(
            job_id=job_id,
            video_stats=stats,
            detections_data=df,
            analytics_data=analytics.get_analytics_data(),
            xai_frames=xai_frames[:5]  # Include up to 5 XAI frames
        )
            
    except Exception as e:
        print(f'Error processing video: {e}')
        if job_id in processing_progress:
            del processing_progress[job_id]  # Clear progress on error
    finally:
        if 'cap' in locals() and cap is not None:
            cap.release()
        if os.path.exists(video_path):
            os.remove(video_path)

# Initialize detector globally
detector = IntegratedDetectorTracker(
    model_path="yolov10x.pt",
    conf_thresh=0.3
)

@routes.route('/api/process_video', methods=['POST'])
def process_video_stream():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
            
        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({'error': 'No selected file'}), 400
            
        # Generate unique timestamp for job tracking
        job_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save temporary file
        temp_path = upload_dir / f'temp_{job_id}.mp4'
        video_file.save(str(temp_path))
        
        # Process video in chunks
        try:
            cap = cv2.VideoCapture(str(temp_path))
            if not cap.isOpened():
                raise ValueError('Could not open video file')
                
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create output video writer
            output_path = result_dir / f'output_{job_id}.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, 
                                (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
            
            frame_count = 0
            detections_data = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                results = detector.detect_and_track(frame)
                frame_with_viz = frame.copy()
                
                # Draw detections and tracking
                for det in results:
                    if det['class_name'] == 'person':
                        x1, y1, w, h = det['bbox']
                        cv2.rectangle(frame_with_viz, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
                        label = f"Person {det.get('track_id', '')}: {det['confidence']:.2f}"
                        cv2.putText(frame_with_viz, label, (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Write processed frame
                out.write(frame_with_viz)
                
                # Send progress update
                if frame_count % 5 == 0:  # Update every 5 frames
                    progress = int((frame_count / total_frames) * 100)
                    socketio.emit('processing_progress', {
                        'job_id': job_id,
                        'progress': progress,
                        'frame_count': frame_count,
                        'total_frames': total_frames
                    })
                
                frame_count += 1
            
            # Cleanup
            cap.release()
            out.release()
            
            return jsonify({
                'success': True,
                'job_id': job_id,
                'output_video': f'/results/output_{job_id}.mp4'
            })
            
        finally:
            if 'cap' in locals():
                cap.release()
            if 'out' in locals():
                out.release()
            if temp_path.exists():
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@routes.route('/analysis')
def analysis():
    try:
        # Get list of result images and videos
        base_dir = Path(__file__).parent.resolve()
        result_images = {
            'detection': [f for f in os.listdir(base_dir / 'static' / 'images' / 'results') 
                         if f.startswith('detection_')],
            'xai': [f for f in os.listdir(base_dir / 'static' / 'images' / 'results') 
                    if f.startswith('xai_')],
            'tracking': [f for f in os.listdir(base_dir / 'static' / 'images' / 'results') 
                        if f.startswith('tracking_')]
        }

        # Add video paths
        demo_videos = {
            'video1': 'videos/demo/video1.mp4',
            'video2': 'videos/demo/video2.mp4'
        }

        return render_template('analysis.html', 
                             result_images=result_images,
                             demo_videos=demo_videos)
    except Exception as e:
        current_app.logger.error(f"Error in analysis route: {str(e)}")
        return "An error occurred while loading the analysis page", 500

@routes.route('/results/<path:filename>')
def serve_result(filename):
    try:
        return send_file(
            str(result_dir / filename),
            mimetype='video/mp4'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404