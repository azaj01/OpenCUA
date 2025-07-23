from typing import Literal

from pydantic import Field

from ...schema.action.action import Action


class CodeAction(Action):
    language: Literal["bash", "python"] = Field(..., description="The language of the code to execute")
    content: str = Field(..., description="The code to execute")
    description: str = Field(..., description="The description/thought provided for the action")
