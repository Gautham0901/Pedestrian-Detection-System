import logging
from typing import List, Tuple, Dict, Any
import os

from .detector import ObjectDetector
from configs.model_config import (
    YOLO_MODEL_PATH,
    YOLO_CONF_THRESH
)

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        """Initialize model manager with YOLO detector"""
        self.detector = ObjectDetector(
            model_path=YOLO_MODEL_PATH,
            conf_thresh=YOLO_CONF_THRESH
        )
        logger.info("Initialized YOLO detector")

    def detect(self, frame) -> List[Dict[str, Any]]:
        """Perform detection on the input frame"""
        return self.detector.detect(frame)

    @property
    def target_classes(self) -> Dict[int, Dict[str, Any]]:
        """Get the target classes from the current detector"""
        return self.detector.target_classes