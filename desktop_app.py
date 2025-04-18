import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import torch
from PIL import Image, ImageTk
from modules.detector import ObjectDetector
from modules.xai import XAIVisualizer
from modules.integrated_detector import IntegratedDetectorTracker
from datetime import datetime
from modules.metrics import DetectionMetrics
from modules.styles import DashboardStyle, MetricCard, DetectionChart

class PedestrianDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pedestrian Detection System")
        self.root.state('zoomed')  # Start maximized
        
        # Detection settings
        self.show_boxes = tk.BooleanVar(value=True)
        self.show_labels = tk.BooleanVar(value=True)
        self.show_xai = tk.BooleanVar(value=True)
        self.confidence_threshold = tk.DoubleVar(value=0.3)
        
        # Initialize detector and metrics
        self.initialize_detector()
        self.metrics = DetectionMetrics()
        
        # Setup UI and styles
        DashboardStyle.configure_style()
        self.setup_ui()
        
        # Video variables
        self.cap = None
        self.is_webcam = False
        self.is_playing = False
        self.frame_buffer = []
        self.buffer_size = 10
        self.current_frame_index = 0
        self.total_frames = 0
        self.fps = 0
        self.last_frame_time = 0
        
    def initialize_detector(self):
        try:
            project_root = Path(__file__).parent
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.detector = IntegratedDetectorTracker(
                model_path=str(project_root / "models" / "yolov10x.pt"),
                conf_thresh=0.3
            )
            self.detector.detector.model = self.detector.detector.model.to(device)
            self.xai_visualizer = XAIVisualizer(self.detector.detector.model)
            
        except Exception as e:
            print(f"Error initializing detector: {e}")
            sys.exit(1)

    def setup_ui(self):
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Control Panel (Left Side)
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Control Panel")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Source selection buttons
        ttk.Button(self.control_frame, text="Open Video File", 
                  command=self.open_video_file).pack(pady=5, padx=5, fill=tk.X)
        ttk.Button(self.control_frame, text="Open Webcam", 
                  command=self.open_webcam).pack(pady=5, padx=5, fill=tk.X)
        
        # Playback controls
        playback_frame = ttk.LabelFrame(self.control_frame, text="Playback Controls")
        playback_frame.pack(pady=5, padx=5, fill=tk.X)
        
        ttk.Button(playback_frame, text="Play/Pause", 
                  command=self.toggle_playback).pack(pady=2, padx=5, fill=tk.X)
        ttk.Button(playback_frame, text="Stop", 
                  command=self.stop_video).pack(pady=2, padx=5, fill=tk.X)
        ttk.Button(playback_frame, text="Restart", 
                  command=self.restart_video).pack(pady=2, padx=5, fill=tk.X)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(playback_frame, variable=self.progress_var,
                                          mode='determinate')
        self.progress_bar.pack(pady=5, padx=5, fill=tk.X)
        
        # Dashboard metrics
        metrics_frame = ttk.Frame(self.control_frame, style='Dashboard.TFrame')
        metrics_frame.pack(pady=10, padx=5, fill=tk.X)
        
        # Metric cards
        # self.fps_card = MetricCard(metrics_frame, "FPS")
        # self.fps_card.pack(pady=5, fill=tk.X)
        
        self.detections_card = MetricCard(metrics_frame, "Detections")
        self.detections_card.pack(pady=5, fill=tk.X)
        
        self.confidence_card = MetricCard(metrics_frame, "Avg Confidence", "0%")
        self.confidence_card.pack(pady=5, fill=tk.X)
        
        # Detection trend chart
        chart_frame = ttk.LabelFrame(self.control_frame, text="Detection Trend", style='Dashboard.TFrame')
        chart_frame.pack(pady=10, padx=5, fill=tk.X)
        self.detection_chart = DetectionChart(chart_frame)
        
        # Detection settings
        settings_frame = ttk.LabelFrame(self.control_frame, text="Detection Settings")
        settings_frame.pack(pady=10, padx=5, fill=tk.X)
        
        ttk.Checkbutton(settings_frame, text="Show Boxes", 
                       variable=self.show_boxes).pack(pady=2, padx=5, anchor=tk.W)
        ttk.Checkbutton(settings_frame, text="Show Labels", 
                       variable=self.show_labels).pack(pady=2, padx=5, anchor=tk.W)
        ttk.Checkbutton(settings_frame, text="Show XAI", 
                       variable=self.show_xai).pack(pady=2, padx=5, anchor=tk.W)
        
        # Confidence threshold slider and value label
        ttk.Label(settings_frame, text="Confidence Threshold").pack(pady=2, padx=5)
        self.conf_value_label = ttk.Label(settings_frame, text="0.3")
        self.conf_value_label.pack(pady=2, padx=5)
        conf_slider = ttk.Scale(settings_frame, from_=0.0, to=1.0, 
                             variable=self.confidence_threshold,
                             command=self.update_conf_label)
        conf_slider.pack(pady=2, padx=5, fill=tk.X)
        
        # Video display area (Right Side)
        self.video_frame = ttk.Label(self.main_frame)
        self.video_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        self.status_bar = ttk.Label(self.root, textvariable=self.status_var, 
                                  relief=tk.SUNKEN)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_video_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if file_path:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.is_webcam = False
            self.is_playing = True
            self.update_frame()

    def open_webcam(self):
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.is_webcam = True
        self.is_playing = True
        self.update_frame()

    def update_conf_label(self, value):
        self.conf_value_label.config(text=f"{float(value):.2f}")

    def process_frame(self, frame):
        settings = {
            'show_boxes': self.show_boxes.get(),
            'show_labels': self.show_labels.get(),
            'show_xai': self.show_xai.get()
        }
        
        try:
            processed_frame = frame.copy()
            results = self.detector.detect_and_track(frame)
            
            # Process each detection
            for det in results:
                bbox = det['bbox']
                x1, y1, w, h = map(int, bbox)
                x2, y2 = x1 + w, y1 + h
                
                # Draw bounding box if enabled
                if settings.get('show_boxes', True):
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), det['color'], 2)
                
                # Draw label if enabled
                if settings.get('show_labels', True):
                    track_id = det.get('track_id', None)
                    label = f"{det['class_name']}: {det['confidence']:.2f}"
                    if track_id is not None:
                        label += f" ID: {track_id}"
                    cv2.putText(processed_frame, label, (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
                # Generate and overlay XAI visualization
                if settings.get('show_xai', True):
                    processed_frame = self.xai_visualizer.generate_saliency_map(processed_frame.copy(), det)
            
            return processed_frame
        except Exception as e:
            print(f"Error processing frame: {e}")
            return frame

    def update_frame(self):
        if self.cap is not None and self.is_playing:
            # Calculate and update FPS
            current_time = datetime.now().timestamp()
            if self.last_frame_time:
                self.fps = 1 / (current_time - self.last_frame_time)
                self.metrics.update_fps(self.fps)
                self.fps_card.update_value(f"{self.fps:.1f}")
            self.last_frame_time = current_time
            
            ret, frame = self.cap.read()
            if ret:
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Update metrics and charts
                results = self.detector.detect_and_track(frame)
                self.metrics.update(results)
                
                # Update metric cards
                self.detections_card.update_value(str(len(results)))
                self.confidence_card.update_value(f"{self.metrics.avg_confidence:.1%}")
                
                # Update detection trend chart
                self.detection_chart.update(len(results))
                
                # Convert to PhotoImage
                image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                
                # Get the current size of the video frame widget
                frame_width = self.video_frame.winfo_width()
                frame_height = self.video_frame.winfo_height()
                
                if frame_width > 1 and frame_height > 1:  # Ensure valid dimensions
                    # Calculate aspect ratio
                    img_ratio = image.width / image.height
                    frame_ratio = frame_width / frame_height
                    
                    if img_ratio > frame_ratio:
                        # Width limited
                        new_width = frame_width
                        new_height = int(frame_width / img_ratio)
                    else:
                        # Height limited
                        new_height = frame_height
                        new_width = int(frame_height * img_ratio)
                    
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image=image)
                
                # Update display
                self.video_frame.configure(image=photo)
                self.video_frame.image = photo
                
                # Update progress bar for video files
                if not self.is_webcam:
                    current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                    total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    progress = (current_frame / total_frames) * 100
                    self.progress_var.set(progress)
                
                # Schedule next update with dynamic interval based on target FPS
                target_fps = 30  # Desired FPS
                interval = max(1, int(1000 / target_fps))  # Convert to milliseconds
                self.root.after(interval, self.update_frame)
            elif not self.is_webcam:
                # Video finished
                self.is_playing = False
                self.status_var.set("Video ended")
            else:
                self.is_playing = False
                self.status_var.set("Video ended")

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.status_var.set("Playing")
            self.update_frame()
        else:
            self.status_var.set("Paused")
    
    def stop_video(self):
        self.is_playing = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status_var.set("Stopped")
        self.progress_var.set(0)
        self.video_frame.configure(image='')
    
    def restart_video(self):
        if self.cap is not None and not self.is_webcam:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.is_playing = True
            self.status_var.set("Playing")
            self.update_frame()
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = PedestrianDetectionApp(root)
    app.run()