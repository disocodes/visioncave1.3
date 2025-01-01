import numpy as np
import torch
from typing import List, Tuple, Dict, Any
from .detection_api import BaseDetector

class YOLOv9Detector(BaseDetector):
    """YOLOv9 detector implementation"""
    
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize YOLOv9 detector
        
        Args:
            model_path: Path to YOLOv9 model weights
            device: Device to run inference on ("cuda" or "cpu")
        """
        super().__init__()
        self.device = device
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str) -> torch.nn.Module:
        """Load YOLOv9 model"""
        model = torch.hub.load('WongKinYiu/yolov9', 'custom', model_path)
        model.to(self.device)
        model.eval()
        return model
        
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform object detection on a frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            
        Returns:
            List of detections, each containing:
                - label: str, class label
                - score: float, confidence score
                - box: list, [x1, y1, x2, y2] coordinates
        """
        # Convert BGR to RGB
        frame_rgb = frame[..., ::-1]
        
        # Inference
        results = self.model(frame_rgb)
        
        # Process results
        detections = []
        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = box
            detection = {
                "label": results.names[int(cls)],
                "score": float(conf),
                "box": [float(x1), float(y1), float(x2), float(y2)]
            }
            detections.append(detection)
            
        return detections
        
    @property
    def name(self) -> str:
        return "yolov9"
