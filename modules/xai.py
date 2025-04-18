import torch
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class XAIVisualizer:
    def __init__(self, model):
        """Initialize XAI visualizer with the model"""
        self.model = model
        # Get the last convolutional layer
        self.target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.target_layer = module
        if self.target_layer is None:
            logger.warning("No convolutional layer found for visualization")
            
    def generate_saliency_map(self, image, detection):
        """Generate a lightweight saliency map for a detected object"""
        try:
            # Ensure image is numpy array
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            # Extract bbox from detection
            if isinstance(detection, (list, tuple)):
                bbox = detection[0] if len(detection) > 0 else None
            else:
                bbox = detection.get('bbox', None)
                
            if bbox is None:
                logger.error("Invalid detection format: bbox not found")
                return image

            # Get bounding box coordinates
            if isinstance(bbox, (list, tuple)):
                x1, y1, w, h = map(int, bbox)
            else:
                x1, y1, w, h = map(int, [bbox.get(k, 0) for k in ['x', 'y', 'width', 'height']])
            x2, y2 = x1 + w, y1 + h
            
            # Ensure coordinates are within image bounds
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            
            if y2 <= y1 or x2 <= x1:
                return image

            # Extract ROI
            roi = image[y1:y2, x1:x2]
            
            # Generate simple gradient-based saliency
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gradient_x = cv2.Sobel(roi_gray, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(roi_gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Normalize gradient magnitude
            gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            
            # Apply colormap to gradient magnitude
            heatmap = cv2.applyColorMap(gradient_magnitude.astype(np.uint8), cv2.COLORMAP_JET)
            
            # Blend with original ROI
            blended = cv2.addWeighted(roi, 0.7, heatmap, 0.3, 0)
            
            # Place blended ROI back in the image
            image[y1:y2, x1:x2] = blended
            
            return image
            
        except Exception as e:
            logger.error(f"Error generating saliency map: {str(e)}")
            return image