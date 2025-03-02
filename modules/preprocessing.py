import cv2
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.preprocessing_config = config['preprocessing']
        
    def process_video(self, video_path):
        """Process video frames according to preprocessing configuration"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            processed_frame = self.preprocess_frame(frame)
            frames.append(processed_frame)
            
        cap.release()
        return frames
        
    def preprocess_frame(self, frame):
        """Apply preprocessing steps to a single frame"""
        if frame is None:
            return None
            
        # Resize if configured
        if self.preprocessing_config.get('resize', False):
            frame = cv2.resize(frame, (
                self.preprocessing_config['width'],
                self.preprocessing_config['height']
            ))
            
        # Normalize if configured
        if self.preprocessing_config.get('normalize', False):
            frame = frame / 255.0
            
        return frame 