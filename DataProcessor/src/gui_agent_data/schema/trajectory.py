from typing import Literal

from pydantic import BaseModel

from ..schema.action.api import ApiAction
from ..schema.action.code import CodeAction
from ..schema.action.gui import GUIAction
from ..schema.action.message import MessageAction
from ..schema.observation.image import ImageObservation
from ..schema.observation.text import TextObservation
from ..schema.observation.multimodal import MultimodalObservation  # Add this import


class Trajectory(BaseModel):
    task_id: str
    example_id: str | None = None
    type: Literal["grounding", "end2end"]
    content: list[GUIAction | ApiAction | CodeAction | MessageAction | TextObservation | ImageObservation | MultimodalObservation]