import cv2
import numpy as np
from typing import List, Dict, Tuple
import logging
from .tracker import TrackingInfo

logger = logging.getLogger(__name__)


class Visualizer:
    def __init__(self, class_names: List[str]):
        """
        Initialize visualizer
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(class_names), 3))
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_tracks(self, frame, tracks: List[TrackingInfo],
                    show_boxes=True, show_labels=True, show_trails=True,
                    blur_class_id=None) -> np.ndarray:
        """
        Draw tracking visualization on frame
        Args:
            frame: Input frame
            tracks: List of TrackingInfo objects
            show_boxes: Whether to draw bounding boxes
            show_labels: Whether to draw labels
            show_trails: Whether to draw motion trails
            blur_class_id: Class ID to apply privacy blur
        Returns:
            Frame with visualizations
        """
        try:
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                class_id = track.class_id
                track_id = track.track_id

                # Get color for this class
                color = self.colors[class_id % len(self.colors)]
                B, G, R = map(int, color)

                # Draw bounding box
                if show_boxes:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (B, G, R), 2)

                # Draw label
                if show_labels:
                    text = f"#{track_id} {self.class_names[class_id]}"
                    cv2.rectangle(frame, (x1, y1 - 20), (x1 + len(text) * 12, y1), (B, G, R), -1)
                    cv2.putText(frame, text, (x1 + 5, y1 - 8),
                                self.font, 0.5, (255, 255, 255), 2)

                # Apply privacy blur if needed
                if blur_class_id is not None and class_id == blur_class_id:
                    if 0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]:
                        frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                            frame[y1:y2, x1:x2], (99, 99), 3)

            return frame

        except Exception as e:
            logger.error(f"Error drawing tracks: {str(e)}")
            return frame

    def draw_analytics(self, frame, fps: float, track_count: int,
                       additional_info: Dict = None) -> np.ndarray:
        """
        Draw analytics information on frame
        Args:
            frame: Input frame
            fps: Current FPS
            track_count: Number of active tracks
            additional_info: Additional information to display
        Returns:
            Frame with analytics
        """
        try:
            # Draw FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        self.font, 1, (0, 255, 0), 2)

            # Draw track count
            cv2.putText(frame, f"Tracks: {track_count}", (10, 60),
                        self.font, 1, (0, 255, 0), 2)

            # Draw additional info if provided
            if additional_info:
                y_pos = 90
                for key, value in additional_info.items():
                    cv2.putText(frame, f"{key}: {value}", (10, y_pos),
                                self.font, 1, (0, 255, 0), 2)
                    y_pos += 30

            return frame

        except Exception as e:
            logger.error(f"Error drawing analytics: {str(e)}")
            return frame