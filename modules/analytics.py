import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from .tracker import TrackingInfo
import logging
import cv2

logger = logging.getLogger(__name__)


class Analytics:
    def __init__(self, frame_size: Tuple[int, int]):
        """
        Initialize analytics module
        Args:
            frame_size: (height, width) of video frame
        """
        self.frame_size = frame_size
        self.reset_analytics()

        # Initialize heatmap
        self.heatmap = np.zeros(frame_size, dtype=np.float32)
        self.heatmap_max = 100  # Maximum value for normalization

        # Initialize zones
        self.zones = {}  # Format: {zone_name: [(x1,y1), (x2,y2)]}
        self.zone_counts = defaultdict(int)

        # Track history
        self.track_history = defaultdict(list)
        self.track_times = defaultdict(lambda: {'start': None, 'end': None})
        self.completed_tracks = []

        # Direction counting
        self.direction_counts = defaultdict(int)

    def reset_analytics(self):
        """Reset all analytics counters"""
        self.total_tracks = 0
        self.current_count = 0
        self.peak_count = 0
        self.class_counts = defaultdict(int)
        self.dwell_times = defaultdict(float)
        self.entry_times = {}
        self.total_tracked = 0  # Total number of unique tracks
        self.avg_dwell_time = 0

    def add_zone(self, name: str, coordinates: List[Tuple[int, int]]):
        """Add a monitoring zone"""
        self.zones[name] = coordinates

    def update(self, tracks: List[TrackingInfo]):
        """
        Update analytics with new tracking information
        Args:
            tracks: List of TrackingInfo objects
        """
        try:
            # Update current count
            self.current_count = len(tracks)
            self.peak_count = max(self.peak_count, self.current_count)

            # Update total tracked
            current_ids = set(track.track_id for track in tracks)
            self.total_tracked = len(set(self.track_history.keys()) | current_ids)

            # Process each track
            for track in tracks:
                # Update class counts
                self.class_counts[track.class_id] += 1

                # Update entry times
                if track.track_id not in self.entry_times:
                    self.entry_times[track.track_id] = track.timestamp

                # Update dwell time
                entry_time = self.entry_times[track.track_id]
                dwell_time = (track.timestamp - entry_time).total_seconds()
                self.dwell_times[track.track_id] = dwell_time

                # Update heatmap
                x, y = map(int, track.center)
                if 0 <= x < self.frame_size[1] and 0 <= y < self.frame_size[0]:
                    self.heatmap[y, x] += 1

                # Check zone presence
                self._check_zones(track)

                # Update track history and times
                track_id = track.track_id
                self.track_history[track_id].append(track.bbox)

                # Initialize start time if new track
                if self.track_times[track_id]['start'] is None:
                    self.track_times[track_id]['start'] = len(self.track_history[track_id])

                # Update end time
                self.track_times[track_id]['end'] = len(self.track_history[track_id])

            # Calculate average dwell time
            dwell_times = []
            for track_id, times in self.track_times.items():
                if times['start'] is not None and times['end'] is not None:
                    dwell_time = times['end'] - times['start']
                    dwell_times.append(dwell_time)

            if dwell_times:
                self.avg_dwell_time = np.mean(dwell_times)

        except Exception as e:
            logger.error(f"Error updating analytics: {str(e)}")

    def _check_zones(self, track: TrackingInfo):
        """Check if track is in any monitoring zone"""
        x, y = map(int, track.center)
        point = np.array([x, y])

        for zone_name, zone_coords in self.zones.items():
            polygon = np.array(zone_coords)
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                self.zone_counts[zone_name] += 1

    def get_heatmap(self) -> np.ndarray:
        """Get normalized heatmap"""
        normalized = np.clip(self.heatmap / self.heatmap_max, 0, 1)
        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8),
                                    cv2.COLORMAP_JET)
        return colored

    def get_analytics_data(self) -> Dict:
        """Get current analytics data"""
        return {
            'current_count': self.current_count,
            'peak_count': self.peak_count,
            'class_counts': dict(self.class_counts),
            'zone_counts': dict(self.zone_counts),
            'avg_dwell_time': self.avg_dwell_time,
            'total_tracked': self.total_tracked,
            'direction_counts': self.direction_counts
        }

    def reset(self):
        """Reset analytics"""
        self.current_count = 0
        self.peak_count = 0
        self.total_tracked = 0
        self.avg_dwell_time = 0
        self.track_history.clear()
        self.track_times.clear()
        self.completed_tracks.clear()
        self.direction_counts.clear()