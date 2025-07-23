from pydantic import BaseModel, Field
from typing import Optional

from .observation import Observation
from .image import ImageObservation
from .axtree import AXTreeObservation  # Add this import


class MultimodalObservation(Observation):
    class_: str = Field("multimodal_observation", description="The class of the observation")
    image_observation: Optional[ImageObservation] = Field(None, description="The image part of the observation")
    axtree_observation: Optional[AXTreeObservation] = Field(None, description="The accessibility tree part of the observation")
    source: str = Field(..., description="The source of the observation (e.g. 'user')")