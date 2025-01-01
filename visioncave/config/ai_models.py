from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class DetectorConfig(BaseModel):
    """Configuration for object detectors"""
    type: str = Field(..., description="Type of detector (yolov9, tensorflow, etc)")
    model_path: str = Field(..., description="Path to model weights")
    device: str = Field("cuda", description="Device to run inference on (cuda/cpu)")
    confidence_threshold: float = Field(0.5, description="Detection confidence threshold")
    
class VLMConfig(BaseModel):
    """Configuration for Vision-Language Model"""
    enabled: bool = Field(True, description="Enable VLM features")
    model_name: str = Field(
        "openai/clip-vit-large-patch14",
        description="Name/path of the vision-language model"
    )
    device: str = Field("cuda", description="Device to run inference on (cuda/cpu)")
    embedding_batch_size: int = Field(32, description="Batch size for computing embeddings")
    
class AIModelsConfig(BaseModel):
    """Configuration for AI models"""
    detector: DetectorConfig = Field(..., description="Object detector configuration")
    vlm: Optional[VLMConfig] = Field(None, description="Vision-Language Model configuration")
    
    class Config:
        extra = "allow"
