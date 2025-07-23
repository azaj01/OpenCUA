from pydantic import Field

from .observation import Observation


class TextObservation(Observation):
    class_: str = Field("text_observation", description="The class of the observation")
    content: str = Field(..., description="A textual observation")
    source: str = Field(..., description="The source of the observation (e.g. 'user')")
