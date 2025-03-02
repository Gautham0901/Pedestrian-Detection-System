import numpy as np
import cv2
import torch
import logging
from lime import lime_image
from skimage.segmentation import mark_boundaries

logger = logging.getLogger(__name__)

class GradCAM:
    """Simple GradCAM implementation if grad-cam package is not available"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
    def get_cam(self, input_tensor, target_category=None):
        # Simple implementation of GradCAM
        feature_maps = self.target_layer(input_tensor)
        if target_category is None:
            target_category = torch.argmax(self.model(input_tensor))
            
        output = self.model(input_tensor)
        output[0, target_category].backward()
        
        weights = torch.mean(self.target_layer.grad, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * feature_maps, dim=1, keepdim=True)
        cam = torch.relu(cam)
        
        return cam.detach().cpu().numpy()

class XAIModule:
    def __init__(self, config):
        self.config = config
        self.xai_config = config['xai']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def analyze(self, detections):
        """Analyze detections using XAI techniques"""
        results = {
            'heatmaps': [],
            'explanations': []
        }
        
        if self.xai_config['show_heatmap']:
            results['heatmaps'] = self.generate_heatmaps(detections)
            
        if self.xai_config['show_lime']:
            results['explanations'] = self.generate_lime_explanations(detections)
            
        return results
        
    def generate_heatmaps(self, detections):
        """Generate Grad-CAM heatmaps for detections"""
        try:
            import torch.nn as nn
            heatmaps = []
            
            for detection in detections:
                bbox = detection[0]
                # Process detection and generate heatmap
                heatmap = self._process_single_heatmap(bbox)
                heatmaps.append(heatmap)
                
            return heatmaps
        except Exception as e:
            logger.error(f"Error generating heatmaps: {e}")
            return []
        
    def generate_lime_explanations(self, detections):
        """Generate LIME explanations for detections"""
        try:
            explanations = []
            explainer = lime_image.LimeImageExplainer()
            
            for detection in detections:
                bbox = detection[0]
                # Process detection and generate LIME explanation
                explanation = self._process_single_lime(bbox, explainer)
                explanations.append(explanation)
                
            return explanations
        except Exception as e:
            logger.error(f"Error generating LIME explanations: {e}")
            return []
            
    def _process_single_heatmap(self, bbox):
        # Implementation for single detection heatmap
        pass
        
    def _process_single_lime(self, bbox, explainer):
        # Implementation for single detection LIME explanation
        pass

class XAIVisualizer:
    def __init__(self, model):
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def generate_heatmap(self, image, track):
        """Generate attention heatmap for tracked object"""
        try:
            if not hasattr(track, 'bbox') or track.bbox is None:
                return image
                
            x1, y1, x2, y2 = map(int, track.bbox)
            roi = image[y1:y2, x1:x2]
            
            if roi.size == 0:
                return image
                
            # Generate heatmap
            heatmap = self._generate_gradcam(roi)
            
            # Apply heatmap to image
            return self._apply_heatmap(image, heatmap, (x1, y1, x2, y2))
            
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            return image
            
    def _generate_gradcam(self, image):
        """Generate GradCAM heatmap"""
        try:
            # Prepare image
            img = cv2.resize(image, (224, 224))
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img = img.float() / 255.0
            img = img.to(self.device)
            
            # Get last convolutional layer
            target_layer = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv2d):
                    target_layer = module
                    
            if target_layer is None:
                raise ValueError("Could not find convolutional layer")
                
            # Generate GradCAM
            grad_cam = GradCAM(self.model, target_layer)
            cam = grad_cam.get_cam(img)
            
            return cam[0, 0]  # Return first channel
            
        except Exception as e:
            logger.error(f"Error in GradCAM generation: {e}")
            return np.zeros((224, 224))
            
    def _apply_heatmap(self, image, heatmap, bbox):
        """Apply heatmap to image"""
        try:
            x1, y1, x2, y2 = bbox
            
            # Resize heatmap to bbox size
            heatmap = cv2.resize(heatmap, (x2-x1, y2-y1))
            
            # Normalize heatmap
            heatmap = np.uint8(255 * heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            
            # Apply heatmap to ROI
            output = image.copy()
            output[y1:y2, x1:x2] = cv2.addWeighted(
                output[y1:y2, x1:x2], 0.7,
                heatmap, 0.3, 0
            )
            
            return output
            
        except Exception as e:
            logger.error(f"Error applying heatmap: {e}")
            return image 