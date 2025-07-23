from pydantic import Field

from ...schema.action.action import Action


class MessageAction(Action):
    content: str = Field(..., description="The message to share with the user")
    description: str | None = Field(None, description="The description/thought provided for the action")
