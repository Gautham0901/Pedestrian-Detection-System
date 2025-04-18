import torch
from ultralytics import YOLO
import logging
import time
import os
import numpy as np
from queue import Queue
from threading import Thread

logger = logging.getLogger(__name__)


class ObjectDetector:
    def __init__(self, model_path="yolov10x.pt", conf_thresh=0.3, batch_size=4):
        """
        Initialize the YOLO detector with batch processing support
        """
        self.conf_thresh = conf_thresh
        self.batch_size = batch_size
        self.frame_buffer = Queue(maxsize=batch_size * 2)
        self.result_buffer = Queue()
        self.processing = True
        
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
        
        # Start processing thread
        self.process_thread = Thread(target=self._process_frames, daemon=True)
        self.process_thread.start()

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
            
            # Enable model optimization
            model.to(self.device)
            if self.device.type == 'cuda':
                model.model.half()  # FP16 for faster inference
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _process_frames(self):
        """Process frames in batches"""
        while self.processing:
            frames = []
            while len(frames) < self.batch_size:
                if not self.frame_buffer.empty():
                    frames.append(self.frame_buffer.get())
                else:
                    if len(frames) > 0:  # Process partial batch
                        break
                    time.sleep(0.001)  # Short sleep if buffer is empty
                    continue
            
            if frames:
                try:
                    # Ensure frames are in the correct format for YOLO processing
                    processed_frames = []
                    for frame in frames:
                        if isinstance(frame, torch.Tensor):
                            # Convert to numpy array and ensure correct format (HWC)
                            frame = frame.cpu().numpy()
                            if frame.shape[0] in [1, 3]:  # If in CHW format
                                frame = np.transpose(frame, (1, 2, 0))
                            if frame.dtype == np.float32 and frame.max() <= 1.0:
                                frame = (frame * 255).astype(np.uint8)
                        processed_frames.append(frame)
                    
                    # Batch inference
                    results = self.model(processed_frames, verbose=False)
                    
                    # Process results
                    for result in results:
                        detections = []
                        if isinstance(result, torch.Tensor):
                            # Handle tensor output
                            boxes = result[:, :4]
                            confidences = result[:, 4]
                            labels = result[:, 5]
                            for box, conf, label in zip(boxes, confidences, labels):
                                class_id = int(label)
                                if class_id not in self.target_classes or conf < self.conf_thresh:
                                    continue
                                x1, y1, x2, y2 = map(int, box.tolist())
                                w, h = x2 - x1, y2 - y1
                                detections.append({
                                    'bbox': [x1, y1, w, h],
                                    'confidence': float(conf),
                                    'class_id': class_id,
                                    'class_name': self.target_classes[class_id]['name'],
                                    'color': self.target_classes[class_id]['color'],
                                    'is_person': class_id == 0
                                })
                        else:
                            # Handle Ultralytics Results object
                            for det in result.boxes:
                                label, confidence, bbox = det.cls, det.conf, det.xyxy[0]
                                class_id = int(label)
                                
                                if class_id not in self.target_classes or confidence < self.conf_thresh:
                                    continue

                                x1, y1, x2, y2 = map(int, bbox)
                                w, h = x2 - x1, y2 - y1
                                
                                detections.append({
                                'bbox': [x1, y1, w, h],
                                'confidence': float(confidence),
                                'class_id': class_id,
                                'class_name': self.target_classes[class_id]['name'],
                                'color': self.target_classes[class_id]['color'],
                                'is_person': class_id == 0
                            })
                        
                        self.result_buffer.put(detections)
                        
                        # Update statistics
                        self.total_detections += len(detections)
                        self.frames_processed += 1
                        if self.frames_processed > 0:
                            self.detection_rate = (self.total_detections / self.frames_processed) * 100
                            
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    for _ in range(len(frames)):
                        self.result_buffer.put([])

    def detect(self, frame):
        """Add frame to processing queue and get results"""
        try:
            # Convert frame to numpy array in correct format
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
                if frame.shape[0] in [1, 3]:  # If in CHW format
                    frame = np.transpose(frame, (1, 2, 0))
                if frame.dtype == np.float32 and frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)

            # Add frame to buffer
            self.frame_buffer.put(frame)
            
            # Get results
            return self.result_buffer.get()

        except Exception as e:
            logger.error(f"Error during detection: {str(e)}")
            return []

    def clear_cache(self):
        """Clear cached results and buffers"""
        while not self.frame_buffer.empty():
            self.frame_buffer.get()
        while not self.result_buffer.empty():
            self.result_buffer.get()

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

    def __del__(self):
        """Cleanup resources"""
        self.processing = False
        if hasattr(self, 'process_thread'):
            self.process_thread.join()