import numpy as np
from sklearn.ensemble import IsolationForest
import logging
import time

logger = logging.getLogger(__name__)

class AnomalyDetector:
    def __init__(self, config):
        self.config = config
        self.model = IsolationForest(random_state=42)
        
    def detect(self, detections):
        """Detect anomalies in object detections"""
        # Extract features from detections
        features = self.extract_features(detections)
        
        # Detect anomalies
        anomaly_scores = self.model.fit_predict(features)
        
        # Process results
        anomalies = self.process_anomalies(detections, anomaly_scores)
        
        return anomalies
        
    def extract_features(self, detections):
        """Extract relevant features for anomaly detection"""
        if not detections:
            return np.array([])
        
        features = []
        for detection in detections:
            # Extract spatial features
            bbox = detection[0]  # [x, y, w, h]
            confidence = detection[1]
            
            # Calculate additional features
            area = bbox[2] * bbox[3]  # width * height
            aspect_ratio = bbox[2] / bbox[3]  # width / height
            center_x = bbox[0] + bbox[2]/2
            center_y = bbox[1] + bbox[3]/2
            
            # Combine features
            feature_vector = [
                area,
                aspect_ratio,
                center_x,
                center_y,
                confidence
            ]
            features.append(feature_vector)
        
        return np.array(features)
        
    def process_anomalies(self, detections, scores):
        """Process anomaly scores and return anomaly information"""
        anomalies = []
        
        for detection, score in zip(detections, scores):
            if score == -1:  # Isolation Forest marks anomalies as -1
                bbox, confidence, class_id = detection
                anomaly = {
                    'bbox': bbox,
                    'confidence': confidence,
                    'class_id': class_id,
                    'anomaly_score': score,
                    'timestamp': time.time()
                }
                anomalies.append(anomaly)
            
        return anomalies 