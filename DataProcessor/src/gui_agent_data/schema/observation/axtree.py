from pydantic import BaseModel, Field
from typing import List, Optional
from .observation import Observation


class AXFrame(BaseModel):
    x: float
    y: float
    w: float
    h: float
    type: str = "kAXValueCGRectType"


class AXNode(BaseModel):
    AXColumns: List['AXNode'] = Field(default_factory=list)
    AXEnabled: Optional[str] = None
    AXFocused: Optional[str] = None
    AXFrame: Optional[AXFrame] = None
    AXRole: Optional[str] = None
    AXRoleDescription: Optional[str] = None
    AXSubrole: Optional[str] = None
    AXHelp: Optional[str] = None
    AXDOMIdentifier: Optional[str] = None
    AXDescription: Optional[str] = None
    AXSelected: Optional[str] = None
    AXTitle: Optional[str] = None
    AXValue: Optional[str] = None
    AXFullScreen: Optional[str] = None


class AXTreeObservation(Observation):
    class_: str = Field("axtree_observation", description="The class of the observation")
    content: AXNode = Field(..., description="The accessibility tree content")
    source: str = Field(..., description="The source of the observation (e.g. 'os')")


# This is needed for the recursive AXNode definition
AXNode.update_forward_refs()