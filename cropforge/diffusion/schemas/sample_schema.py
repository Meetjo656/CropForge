"""
Pydantic Dataset Sample Schema definition for CropForge Diffusion Dataset Generation.
"""

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class DatasetSample(BaseModel):
    """
    Unified dataset sample schema for Stable Diffusion 3.5 LoRA dataset generation.
    """
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    sample_id: str = Field(..., description="Unique identifier for the training sample")
    crop: str = Field(..., description="Target crop name, e.g., Tomato")
    disease: str = Field(..., description="Plant disease or state, e.g., Late Blight")
    severity: str = Field(..., description="Disease severity level: Healthy, Mild, Moderate, Severe, Critical")
    treatment: str = Field(..., description="Applied treatment, e.g., Copper Fungicide")
    days_after_treatment: int = Field(..., description="Days elapsed since treatment application")
    temperature: float = Field(..., description="Ambient temperature in degrees Celsius")
    humidity: float = Field(..., description="Relative ambient humidity percentage")
    input_image: str = Field(..., description="Path or filename of the initial input image")
    target_image: str = Field(..., description="Path or filename of the target image")
    segmentation_mask: str = Field(..., description="Path or filename of the segmentation mask")

    lighting: Optional[str] = Field(None, description="Lighting condition for rendering")
    camera_style: Optional[str] = Field(None, description="Camera or photographic style")
    extra_description: Optional[str] = Field(None, description="Additional context or prompt notes")

    def to_dict(self) -> dict:
        """Convert sample to standard dictionary."""
        return self.model_dump()
