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
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            embedder_gpu=True
        )
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
            # Update tracker
            tracks = self.tracker.update_tracks(detections, frame=frame)

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