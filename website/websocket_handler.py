import asyncio
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from modules.integrated_detector import IntegratedDetectorTracker
from modules.xai import XAIVisualizer

class WebSocketHandler:
    def __init__(self, socketio):
        self.socketio = socketio
        self.detector = IntegratedDetectorTracker(model_path='yolov10x.pt', conf_thresh=0.3)
        self.xai_visualizer = XAIVisualizer(self.detector.detector.model)
        self.processing = False
        self.frame_count = 0
        self.total_detections = 0
        self.current_count = 0
        self.peak_count = 0
        
    def process_frame(self, frame_data):
        try:
            # Decode frame data
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            processed_frame = frame.copy()
            
            # Run detection and tracking
            results = self.detector.detect_and_track(frame)
            
            # Filter for person detections
            person_detections = [det for det in results if det.get('class_name', '') == 'person']
            
            # Update metrics
            self.current_count = len(person_detections)
            self.total_detections += self.current_count
            self.peak_count = max(self.peak_count, self.current_count)
            
            # Calculate average confidence
            avg_confidence = 0
            if person_detections:
                avg_confidence = sum(det['confidence'] for det in person_detections) / len(person_detections)
            
            # Process detections and generate XAI visualizations
            detection_data = []
            for det in person_detections:
                bbox = det['bbox']
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                
                # Draw bounding box with unique color based on track ID
                track_id = det.get('track_id', 0)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label with track ID and confidence
                label = f"Person {track_id}: {det['confidence']:.2f}"
                cv2.putText(processed_frame, label, (x1, y1-10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Store detection data for frontend
                detection_data.append({
                    'track_id': track_id,
                    'confidence': det['confidence'],
                    'bbox': [x1, y1, w, h]
                })
                
                # Generate XAI visualization for high-confidence detections
                if det['confidence'] > 0.5:
                    processed_frame = self.xai_visualizer.generate_saliency_map(processed_frame.copy(), det)
            
            # Encode processed frame
            _, buffer = cv2.imencode('.jpg', processed_frame)
            processed_frame_data = buffer.tobytes()
            
            # Update frame count and emit results
            self.frame_count += 1
            self.socketio.emit('frame_processed', {
                'frame': processed_frame_data,
                'detections': detection_data,
                'metrics': {
                    'total_detections': self.total_detections,
                    'current_count': self.current_count,
                    'peak_count': self.peak_count,
                    'frame_count': self.frame_count,
                    'avg_confidence': float(avg_confidence)
                }
            })
            
        except Exception as e:
            self.socketio.emit('processing_error', {'error': str(e)})
    
    def start_processing(self):
        self.processing = True
        self.frame_count = 0
        self.total_detections = 0
        self.current_count = 0
        self.peak_count = 0
        self.socketio.emit('processing_started')
    
    def stop_processing(self):
        self.processing = False
        self.socketio.emit('processing_stopped')