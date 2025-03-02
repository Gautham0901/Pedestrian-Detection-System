import torch
from ultralytics import YOLO
import logging
import time
import os

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path="yolov8x.pt", conf_thresh=0.3):
        """
        Initialize the YOLO detector
        """
        self.conf_thresh = conf_thresh
        # COCO class IDs for vehicles and pedestrians
        self.target_classes = {
            0: {'name': 'person', 'color': (255, 0, 0)},    # Blue for pedestrians
            2: {'name': 'car', 'color': (0, 255, 0)},       # Green
            3: {'name': 'motorcycle', 'color': (0, 0, 255)}, # Red
            5: {'name': 'bus', 'color': (255, 255, 0)},     # Cyan
            7: {'name': 'truck', 'color': (255, 0, 255)},   # Magenta
            1: {'name': 'bicycle', 'color': (0, 165, 255)}  # Orange
        }
        self.device = self._get_device()
        self.model = self._load_model(model_path)
        self.total_detections = 0
        self.frames_processed = 0
        self.start_time = time.time()
        self.detection_rate = 0.0

    def _get_device(self):
        """Determine the appropriate device (CUDA, MPS, or CPU)"""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info(f"Using {device} for inference")
        return device

    def _load_model(self, model_path):
        """Load and configure the YOLO model"""
        try:
            if not os.path.exists(model_path):
                model = YOLO("yolov10x.pt")  # This will download the model
            else:
                model = YOLO(model_path)
            
            # Ensure model is in eval mode and on correct device
            model.to(self.device)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def detect(self, frame):
        """
        Detect both pedestrians and vehicles in a frame
        """
        try:
            results = self.model(frame, verbose=False)[0]
            detections = []

            for det in results.boxes:
                label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
                class_id = int(label)
                
                # Only process target classes
                if class_id not in self.target_classes:
                    continue

                # Filter by confidence
                if confidence < self.conf_thresh:
                    continue

                x1, y1, x2, y2 = map(int, bbox)
                w, h = x2 - x1, y2 - y1
                
                detections.append({
                    'bbox': [x1, y1, w, h],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': self.target_classes[class_id]['name'],
                    'color': self.target_classes[class_id]['color'],
                    'is_person': class_id == 0  # Flag to identify pedestrians
                })

            # Update statistics
            self.total_detections += len(detections)
            self.frames_processed += 1
            if self.frames_processed > 0:
                self.detection_rate = (self.total_detections / self.frames_processed) * 100

            return detections

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []

    def clear_cache(self):
        """Clear cached results"""
        self.cache = {}

    def get_class_names(self):
        """Return list of class names"""
        return self.model.names

    def get_current_frame(self):
        """Get the current frame from the video source"""
        if hasattr(self, 'cap') and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                return frame
        return None

    def get_fps(self):
        """Calculate the current processing FPS"""
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frames_processed / elapsed_time
        return 0.0