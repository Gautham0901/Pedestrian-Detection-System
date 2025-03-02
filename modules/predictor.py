import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import logging
import time

logger = logging.getLogger(__name__)

class ThreatPredictor:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.model = self._build_model()
        
    def _build_model(self):
        """Build the threat prediction model"""
        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(None, 5)),  # 5 features from anomaly detector
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model
        
    def predict(self, anomalies):
        """Predict future threats based on current anomalies"""
        if not anomalies:
            return []
        
        # Extract features from anomalies
        features = []
        for anomaly in anomalies:
            feature_vector = [
                anomaly['bbox'][2] * anomaly['bbox'][3],  # area
                anomaly['confidence'],
                anomaly['anomaly_score'],
                anomaly['bbox'][0] + anomaly['bbox'][2]/2,  # center_x
                anomaly['bbox'][1] + anomaly['bbox'][3]/2   # center_y
            ]
            features.append(feature_vector)
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        # Make prediction
        predictions = self.model.predict(np.array([features]))
        
        # Process predictions
        threats = []
        for i, prob in enumerate(predictions[0]):
            if prob > self.config['detection']['confidence_threshold']:
                threats.append({
                    'anomaly': anomalies[i],
                    'threat_probability': float(prob),
                    'predicted_time': time.time() + 30  # 30 seconds into future
                })
        
        return threats 