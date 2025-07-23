import random
from abc import abstractmethod

from ..schema.action.gui import GUIAction, GUIElement, PyAutoGUIAction


class GUIActionConstructor:
    def __init__(self):
        pass

    @abstractmethod
    def construct(self) -> GUIAction:
        pass


class ClickConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Click on '{self.element.text}'" if self.element.text else "Click",
            guiactions=[
                PyAutoGUIAction(
                    target=self.element,
                    action_type="click",
                    args={"x": self.element.center[0], "y": self.element.center[1]},
                )
            ],
        )


class DoubleClickConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Double-click on '{self.element.text}'" if self.element.text else "Double-click",
            guiactions=[
                PyAutoGUIAction(
                    target=self.element,
                    action_type="doubleClick",
                    args={"x": self.element.center[0], "y": self.element.center[1]},
                )
            ],
        )


class RightClickConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Right-click on '{self.element.text}'" if self.element.text else "Right-click",
            guiactions=[
                PyAutoGUIAction(
                    target=self.element,
                    action_type="rightClick",
                    args={"x": self.element.center[0], "y": self.element.center[1]},
                )
            ],
        )


class MoveToConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Move to '{self.element.text}'" if self.element.text else "Move to",
            guiactions=[
                PyAutoGUIAction(
                    target=self.element,
                    action_type="moveTo",
                    args={"x": self.element.center[0], "y": self.element.center[1]},
                )
            ],
        )


class SelectConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Drag to select '{self.element.text}'" if self.element.text else "Drag to select",
            guiactions=[
                PyAutoGUIAction(
                    target=self.element,
                    action_type="moveTo",
                    args={"x": self.element.left, "y": self.element.center[1]},
                ),
                PyAutoGUIAction(
                    target=self.element,
                    action_type="dragTo",
                    args={"x": self.element.right, "y": self.element.center[1]},
                ),
            ],
        )


class TypewriteConstructor(GUIActionConstructor):
    def __init__(self, text: str):
        self.text = text

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Type '{self.text}'",
            guiactions=[PyAutoGUIAction(action_type="write", args={"text": self.text}, target=None)],
        )


class ScrollConstructor:
    def __init__(self, dx: int, dy: int, x: int = -1, y: int = -1):
        self.dx = dx
        self.dy = dy
        self.x = x
        self.y = y

    def construct(self) -> GUIAction:
        if self.x == -1 and self.y == -1:
            return GUIAction(
                instruction=f"Scroll by ({self.dx}, {self.dy})",
                guiactions=[PyAutoGUIAction(action_type="scroll", args={"dx": self.dx, "dy": self.dy}, target=None)],
            )
        else:
            return GUIAction(
                instruction=f"Scroll by ({self.dx}, {self.dy}) from ({self.x}, {self.y})",
                guiactions=[
                    PyAutoGUIAction(
                        action_type="scroll", args={"dx": self.dx, "dy": self.dy, "x": self.x, "y": self.y}, target=None
                    )
                ],
            )


class KeypressConstructor:
    def __init__(self, element: GUIElement | None, key: str):
        self.element = element
        self.key = key

    def construct(self) -> GUIAction:
        return GUIAction(
            instruction=f"Press key '{self.key}'",
            guiactions=[PyAutoGUIAction(target=self.element, action_type="press", args={"key": self.key})],
        )


class RandomGUIActionConstructor(GUIActionConstructor):
    def __init__(self, element: GUIElement):
        self.element = element
        self.constructors = [
            ClickConstructor(element),
            DoubleClickConstructor(element),
            RightClickConstructor(element),
            MoveToConstructor(element),
            SelectConstructor(element),
        ]

    def construct(self) -> GUIAction:
        chosen_constructor = random.choice(self.constructors)
        return chosen_constructor.construct()
