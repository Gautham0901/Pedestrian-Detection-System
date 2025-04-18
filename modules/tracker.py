from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
import logging
from dataclasses import dataclass
from typing import List, Tuple, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class TrackingInfo:
    """Data class to store tracking information"""
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    class_id: int
    confidence: float
    center: Tuple[float, float]
    timestamp: datetime


class TrackerModule:
    def __init__(self, max_age=20, n_init=3):
        """
        Initialize DeepSORT tracker
        Args:
            max_age: Maximum number of frames to keep track alive
            n_init: Number of frames needed to confirm a track
        """
        try:
            import torch
            # Ensure PyTorch is properly initialized
            if not torch._C._GLIBCXX_USE_CXX11_ABI:
                torch._C._GLIBCXX_USE_CXX11_ABI = True
            
            self.tracker = DeepSort(
                max_age=max_age,
                n_init=n_init,
                embedder_gpu=False,  # Set to False to avoid GPU-related issues
                max_cosine_distance=0.3,  # Add explicit distance threshold
                nn_budget=None  # Disable budget to prevent memory issues
            )
        except Exception as e:
            logger.error(f"Error initializing DeepSort tracker: {e}")
            # Initialize with a simple dictionary to store basic tracking info
            self.tracker = None
            # Log additional diagnostic information
            logger.info("Falling back to basic tracking mode due to initialization error")
            try:
                import torch.version
                logger.info(f"PyTorch version: {torch.__version__}")
            except:
                pass
        self.track_history: Dict[int, List[TrackingInfo]] = {}
        self.active_tracks: List[TrackingInfo] = []

    def update(self, detections: List, frame) -> List[TrackingInfo]:
        """
        Update tracker with new detections
        Args:
            detections: List of detections [[x1,y1,w,h], confidence, class_id]
            frame: Current frame
        Returns:
            List of TrackingInfo objects
        """
        try:
            # Check if tracker is initialized
            if self.tracker is None:
                logger.warning("Tracker not initialized. Attempting to reinitialize...")
                try:
                    import torch
                    if not torch._C._GLIBCXX_USE_CXX11_ABI:
                        torch._C._GLIBCXX_USE_CXX11_ABI = True
                    
                    self.tracker = DeepSort(
                        max_age=20,
                        n_init=3,
                        embedder_gpu=False,
                        max_cosine_distance=0.3,
                        nn_budget=None
                    )
                except Exception as e:
                    logger.error(f"Failed to reinitialize tracker: {e}")
                    return []
            
            # Ensure detections is a list and not an int or other non-iterable type
            if detections is None or not isinstance(detections, list):
                logger.warning(f"Invalid detections format: {type(detections)}. Expected a list.")
                return []
            
            # Ensure each detection in the list is properly formatted
            valid_detections = []
            for det in detections:
                try:
                    # Handle list format [x,y,w,h,confidence,class_id]
                    if isinstance(det, list) and len(det) >= 5:
                        valid_detections.append(det)
                    # Handle dictionary format from detector.py
                    elif isinstance(det, dict) and 'bbox' in det and 'confidence' in det and 'class_id' in det:
                        x, y, w, h = det['bbox']
                        valid_detections.append([x, y, w, h, det['confidence'], det['class_id']])
                except Exception as e:
                    logger.warning(f"Invalid detection format: {e}")
                    continue
            
            # If no valid detections, return empty list
            if not valid_detections:
                return []
            
            # Convert detections to DeepSORT format
            detection_boxes = np.array([det[:4] for det in valid_detections])
            detection_scores = np.array([det[4] for det in valid_detections])
            detection_classes = np.array([det[5] for det in valid_detections])
            
            # Update tracker with valid detections only
            tracks = self.tracker.update_tracks(valid_detections, frame=frame)

            # Clear active tracks
            self.active_tracks.clear()

            # Process confirmed tracks
            for track in tracks:
                if not track.is_confirmed():
                    continue

                track_id = track.track_id
                ltrb = track.to_ltrb()
                class_id = track.get_det_class()
                confidence = track.get_det_conf()
                x1, y1, x2, y2 = map(int, ltrb)

                # Calculate center point
                center = ((x1 + x2) / 2, (y1 + y2) / 2)

                # Create tracking info
                track_info = TrackingInfo(
                    track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    class_id=class_id,
                    confidence=confidence,
                    center=center,
                    timestamp=datetime.now()
                )

                # Update track history
                if track_id not in self.track_history:
                    self.track_history[track_id] = []
                self.track_history[track_id].append(track_info)

                # Limit history length
                max_history = 30  # Configurable
                if len(self.track_history[track_id]) > max_history:
                    self.track_history[track_id].pop(0)

                self.active_tracks.append(track_info)

            return self.active_tracks

        except Exception as e:
            logger.error(f"Error updating tracker: {str(e)}")
            return []

    def get_track_history(self, track_id: int) -> List[TrackingInfo]:
        """Get history for a specific track"""
        return self.track_history.get(track_id, [])

    def get_active_tracks(self) -> List[TrackingInfo]:
        """Get currently active tracks"""
        return self.active_tracks

    def get_track_count(self) -> int:
        """Get number of active tracks"""
        return len(self.active_tracks)

    def clear_old_tracks(self, max_age_seconds: float = 5.0):
        """Clear tracks older than max_age_seconds"""
        current_time = datetime.now()
        for track_id in list(self.track_history.keys()):
            history = self.track_history[track_id]
            if not history:
                continue

            last_seen = history[-1].timestamp
            age = (current_time - last_seen).total_seconds()

            if age > max_age_seconds:
                del self.track_history[track_id]

    def clear_tracks(self):
        """Clear all tracks"""
        self.track_history = {}
        self.active_tracks = []