import argparse
import cv2
import logging
import numpy as np
from pathlib import Path
from modules.detector import ObjectDetector
from modules.xai import XAIVisualizer

def setup_logging(log_level):
    """Setup logging configuration"""
    numeric_level = getattr(logging, log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

class VideoProcessor:
    def __init__(self):
        self.recording = False
        self.show_boxes = True
        self.show_labels = True
        self.paused = False
        self.frame_count = 0
        self.detection_count = 0

    def create_control_panel(self, frame):
        """Create control panel overlay"""
        panel = np.zeros((60, frame.shape[1], 3), dtype=np.uint8)
        
        # Add button states
        buttons = [
            ("Recording" if self.recording else "Record [R]", (0, 0, 255) if self.recording else (200, 200, 200)),
            ("Boxes [B]: ON" if self.show_boxes else "Boxes [B]: OFF", (0, 255, 0) if self.show_boxes else (200, 200, 200)),
            ("Labels [L]: ON" if self.show_labels else "Labels [L]: OFF", (0, 255, 0) if self.show_labels else (200, 200, 200)),
            ("Paused [Space]" if self.paused else "Playing [Space]", (0, 0, 255) if self.paused else (0, 255, 0)),
            (f"Detections: {self.detection_count}", (255, 255, 255)),
            ("Quit [Q]", (200, 200, 200))
        ]
        
        x_offset = 10
        for text, color in buttons:
            cv2.putText(panel, text, (x_offset, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            x_offset += int(frame.shape[1] / len(buttons))
            
        return panel

    def handle_key(self, key):
        """Handle keyboard controls"""
        if key == ord('r'):
            self.recording = not self.recording
            return "toggle_recording"
        elif key == ord('b'):
            self.show_boxes = not self.show_boxes
        elif key == ord('l'):
            self.show_labels = not self.show_labels
        elif key == ord(' '):
            self.paused = not self.paused
        elif key == ord('q'):
            return "quit"
        return None

def process_video(input_path, output_path, detector, xai_visualizer):
    """Process video file for pedestrian detection"""
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video processor and writer
        processor = VideoProcessor()
        writer = None
        
        while True:
            if not processor.paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect only pedestrians (class_id 0)
                detections = detector.detect(frame)
                pedestrians = [d for d in detections if d['class_id'] == 0]
                processor.detection_count = len(pedestrians)
                
                # Draw detections and XAI visualizations
                if processor.show_boxes or processor.show_labels:
                    for det in pedestrians:
                        bbox = det['bbox']
                        x1, y1, w, h = bbox
                        
                        # Draw bounding box
                        if processor.show_boxes:
                            cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (255, 0, 0), 2)
                        
                        # Draw label
                        if processor.show_labels:
                            label = f"Person: {det['confidence']:.2f}"
                            cv2.putText(frame, label, (x1, y1-10),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                            
                        # Generate and overlay XAI visualization
                        frame = xai_visualizer.generate_saliency_map(frame.copy(), det)
                
                # Add control panel
                control_panel = processor.create_control_panel(frame)
                frame = np.vstack([control_panel, frame])
                
                # Handle recording
                if processor.recording:
                    if writer is None and output_path:
                        fourcc = cv2.VideoWriter_fourcc(*'XVID')
                        writer = cv2.VideoWriter(
                            output_path, 
                            fourcc, 
                            fps, 
                            (frame.shape[1], frame.shape[0])
                        )
                    if writer is not None:
                        writer.write(frame)
            
            # Show frame
            cv2.imshow('Pedestrian Detection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            action = processor.handle_key(key)
            
            if action == "quit":
                break
                
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        raise
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Pedestrian Detection System')
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input video file')
    parser.add_argument('--output', type=str, default='output.avi',
                      help='Path to output video file')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                      help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Initialize detector and XAI visualizer
        detector = ObjectDetector(
            model_path="yolov10x.pt",
            conf_thresh=0.3
        )
        xai_visualizer = XAIVisualizer(detector.model)  
        
        # Process video
        process_video(
            args.input,
            args.output,
            detector,
            xai_visualizer
        )
        
        logging.info("Video processing completed successfully")
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())