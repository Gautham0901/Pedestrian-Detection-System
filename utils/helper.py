import cv2
import numpy as np
from typing import Tuple, List
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


def load_video_stream(source: str) -> Tuple[cv2.VideoCapture, dict]:
    """
    Initialize video capture and get properties
    Args:
        source: Path to video file, camera index, or URL
    Returns:
        VideoCapture object and dictionary of video properties
    """
    try:
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

        props = {
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }

        return cap, props

    except Exception as e:
        logger.error(f"Error initializing video capture: {str(e)}")
        raise


def save_results(results: dict, output_path: str):
    """Save tracking results to JSON file"""
    try:
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection)