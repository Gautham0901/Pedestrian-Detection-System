import numpy as np
from datetime import datetime

class DetectionMetrics:
    def __init__(self):
        self.total_detections = 0
        self.frames_processed = 0
        self.avg_confidence = 0
        self.detection_history = []
        self.confidence_history = []
        self.fps_history = []
        self.last_update_time = datetime.now()
        
    def update(self, detections):
        self.frames_processed += 1
        num_detections = len(detections)
        self.total_detections += num_detections
        self.detection_history.append(num_detections)
        
        # Update confidence metrics
        if detections:
            confidences = [det['confidence'] for det in detections]
            avg_conf = sum(confidences) / len(confidences)
            self.confidence_history.append(avg_conf)
            self.avg_confidence = np.mean(self.confidence_history)
        else:
            self.confidence_history.append(0)
            
        # Limit history length to prevent memory issues
        max_history = 1000
        if len(self.detection_history) > max_history:
            self.detection_history = self.detection_history[-max_history:]
            self.confidence_history = self.confidence_history[-max_history:]
            self.fps_history = self.fps_history[-max_history:]
    
    def update_fps(self, fps):
        self.fps_history.append(fps)
        
    @property
    def avg_detections_per_frame(self):
        return self.total_detections / max(1, self.frames_processed)
    
    @property
    def peak_detections(self):
        return max(self.detection_history) if self.detection_history else 0
    
    @property
    def current_fps(self):
        return self.fps_history[-1] if self.fps_history else 0
    
    @property
    def avg_fps(self):
        return np.mean(self.fps_history) if self.fps_history else 0