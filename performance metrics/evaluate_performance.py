import time
import numpy as np
from ultralytics import YOLO
import supervision as sv
import cv2
from pathlib import Path
import psutil
import json
from tqdm import tqdm

class PerformanceEvaluator:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.video_path = video_path
        self.metrics = {
            'inference_times': [],
            'fps': [],
            'memory_usage': [],
            'frames_processed': 0,
            'total_detections': 0,
            'detections_per_frame': []  # Add this to track per-frame detections
        }
    
    def evaluate_model(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print("Starting performance evaluation...")
        for _ in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                break
                
            # Measure inference time
            inference_time, results = self.measure_inference_time(frame)
            self.metrics['inference_times'].append(inference_time)
            self.metrics['fps'].append(1.0 / inference_time)
            
            # Measure memory usage
            self.metrics['memory_usage'].append(self.measure_memory_usage())
            
            # Count detections for this frame
            num_detections = len(results[0].boxes)
            self.metrics['frames_processed'] += 1
            self.metrics['total_detections'] += num_detections
            self.metrics['detections_per_frame'].append(num_detections)  # Store per-frame detections
        
        cap.release()
    
    def measure_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    def measure_inference_time(self, frame):
        """Measure inference time for a single frame"""
        try:
            start_time = time.time()
            results = self.model(frame, verbose=False)  # Disable verbose output for cleaner logs
            end_time = time.time()
            return end_time - start_time, results
        except Exception as e:
            print(f"Error during inference: {str(e)}")
            raise
    
    def calculate_average_metrics(self):
        avg_metrics = {
            'average_inference_time': np.mean(self.metrics['inference_times']),
            'average_fps': np.mean(self.metrics['fps']),
            'average_memory_usage': np.mean(self.metrics['memory_usage']),
            'total_frames_processed': self.metrics['frames_processed'],
            'average_detections_per_frame': self.metrics['total_detections'] / self.metrics['frames_processed'] if self.metrics['frames_processed'] > 0 else 0
        }
        return avg_metrics
    
    def save_metrics(self, output_path):
        avg_metrics = self.calculate_average_metrics()
        
        report = {
            'average_metrics': avg_metrics,
            'detailed_metrics': self.metrics
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print("\nPerformance Report:")
        print(f"Average Inference Time: {avg_metrics['average_inference_time']:.4f} seconds")
        print(f"Average FPS: {avg_metrics['average_fps']:.2f}")
        print(f"Average Memory Usage: {avg_metrics['average_memory_usage']:.2f} MB")
        print(f"Total Frames Processed: {avg_metrics['total_frames_processed']}")
        print(f"Average Detections per Frame: {avg_metrics['average_detections_per_frame']:.2f}")

def main():
    import torch
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: Running on CPU. Performance will be limited.")
    
    model_path = r"models\yolov10x.pt"
    video_path = r"videos\test.mp4"
    output_path = "metrics_report.json"
    
    try:
        evaluator = PerformanceEvaluator(model_path, video_path)
        evaluator.evaluate_model()
        evaluator.save_metrics(output_path)
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    main()