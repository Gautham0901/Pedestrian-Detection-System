import torch
import numpy as np
from .detector import ObjectDetector
from .sort import Sort

class IntegratedDetectorTracker:
    def __init__(self, model_path, conf_thresh=0.3, max_age=20, min_hits=3, iou_threshold=0.3):
        self.detector = ObjectDetector(model_path, conf_thresh)
        self.tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        
    def detect_and_track(self, frame):
        # Run detection
        detections = self.detector.detect(frame)
        pedestrians = [d for d in detections if d['class_id'] == 0]
        
        # Prepare tracking input
        if len(pedestrians) > 0:
            boxes = np.array([d['bbox'] for d in pedestrians])
            confidences = np.array([d['confidence'] for d in pedestrians])
            
            # Convert to x1,y1,x2,y2 format
            boxes_xyxy = np.zeros_like(boxes)
            boxes_xyxy[:, 0] = boxes[:, 0]
            boxes_xyxy[:, 1] = boxes[:, 1]
            boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]
            
            # Update tracker
            tracks = self.tracker.update(boxes_xyxy, confidences)
            
            # Combine results
            results = []
            for i, det in enumerate(pedestrians):
                result = det.copy()
                # Find matching track
                for track in tracks:
                    if self._iou(boxes_xyxy[i], track[:4]) > 0.5:
                        result['track_id'] = int(track[4])
                        break
                results.append(result)
            return results
        return []
    
    def _iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection)