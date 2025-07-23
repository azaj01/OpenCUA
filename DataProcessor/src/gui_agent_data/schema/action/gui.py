from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from .action import Action


class GUIElement(BaseModel):
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        None, description="The normalized bounding box of the element (x1, y1, x2, y2)"
    )
    pixel_bbox: Optional[Tuple[int, int, int, int]] = Field(
        None, description="The pixel bounding box of the element (x1, y1, x2, y2)"
    )
    image_size: Tuple[int, int] = Field(..., description="The size of the image (width, height)")
    text: Optional[str] = Field(None, description="The text content of the element")

    @model_validator(mode="before")
    @classmethod
    def check_bbox_or_pixel_bbox(cls, values):
        if values.get("bbox") is None and values.get("pixel_bbox") is None:
            raise ValueError("Either 'bbox' or 'pixel_bbox' must be provided")

        # Calculate pixel_bbox if only bbox is provided
        if values.get("bbox") is not None and values.get("pixel_bbox") is None:
            bbox = values["bbox"]
            image_size = values["image_size"]
            img_width, img_height = image_size
            x1, y1, x2, y2 = bbox
            values["pixel_bbox"] = (
                round(x1 * img_width),
                round(y1 * img_height),
                round(x2 * img_width),
                round(y2 * img_height),
            )

        # Calculate bbox if only pixel_bbox is provided
        elif values.get("pixel_bbox") is not None and values.get("bbox") is None:
            pixel_bbox = values["pixel_bbox"]
            image_size = values["image_size"]
            img_width, img_height = image_size
            x1, y1, x2, y2 = pixel_bbox
            values["bbox"] = (
                x1 / img_width,
                y1 / img_height,
                x2 / img_width,
                y2 / img_height,
            )

        return values

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, v):
        if v is not None:
            if not all(0 <= coord <= 1 for coord in v):
                print(v)
                raise ValueError("All bounding box coordinates must be between 0 and 1")
            if v[0] > v[2] or v[1] > v[3]:
                raise ValueError("Invalid bounding box: x1 must be less than x2, and y1 must be less than y2")
        return v

    @field_validator("pixel_bbox")
    @classmethod
    def validate_pixel_bbox(cls, v, info):
        if v is not None:
            image_size = info.data.get("image_size")
            if image_size:
                img_width, img_height = image_size
                if not all(0 <= v[0] <= v[2] <= img_width and 0 <= v[1] <= v[3] <= img_height):
                    raise ValueError("Pixel bounding box coordinates must be within image dimensions")
            if v[0] > v[2] or v[1] > v[3]:
                raise ValueError("Invalid pixel bounding box: x1 must be less than x2, and y1 must be less than y2")
        return v

    @field_validator("image_size")
    @classmethod
    def validate_image_size(cls, v):
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Image dimensions must be positive")
        return v

    @property
    def normalized_bbox(self) -> Tuple[float, float, float, float]:
        return self.bbox

    @property
    def absolute_bbox(self) -> Tuple[int, int, int, int]:
        return self.pixel_bbox

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def left(self) -> float:
        return self.bbox[0]

    @property
    def top(self) -> float:
        return self.bbox[1]

    @property
    def right(self) -> float:
        return self.bbox[2]

    @property
    def bottom(self) -> float:
        return self.bbox[3]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


import ast
import inspect
from enum import Enum

import pyautogui


class GUIActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "doubleClick"
    RIGHT_CLICK = "rightClick"
    MIDDLE_CLICK = "middleClick"
    MOVE_TO = "moveTo"
    DRAG_TO = "dragTo"
    SCROLL = "scroll"
    HSCROLL = "hscroll"
    WRITE = "write"
    PRESS = "press"
    HOTKEY = "hotkey"


class PyAutoGUIAction(BaseModel):
    action_type: GUIActionType
    target: Optional[GUIElement] = Field(None, description="The target element")
    args: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, description="Arguments for the action")

    @model_validator(mode="after")
    def initialize_args(self):
        if len(self.args) != 0:
            return self
        if self.action_type == GUIActionType.CLICK:
            self.args = {"x": self.target.center[0], "y": self.target.center[1]}
        elif self.action_type == GUIActionType.WRITE:
            self.args = {"message": self.target.text}
        return self

    @field_validator("args")
    @classmethod
    def validate_args(cls, v):
        if v is not None and "x" in v and "y" in v:
            x, y = v["x"], v["y"]
            if not (0 <= x <= 1 and 0 <= y <= 1):
                raise ValueError("x and y coordinates must be between 0 and 1")
        return v

    def to_command(self) -> str:
        def convert_and_round(value):
            if isinstance(value, float):
                return round(value, 4)  # Round to 4 decimal places
            return value

        if isinstance(self.args, list):
            args_str = ", ".join(repr(convert_and_round(v)) for v in self.args)
        elif len(self.args) == 1 and next(iter(self.args.keys())) in ["clicks", "amount", "page"]:
            # For single-argument functions like scroll, just use the value
            args_str = repr(convert_and_round(next(iter(self.args.values()))))
        else:
            args_str = ", ".join(f"{k}={repr(convert_and_round(v))}" for k, v in self.args.items())
        return f"pyautogui.{self.action_type.value}({args_str})"

    @classmethod
    def from_string(cls, action_string: str) -> "PyAutoGUIAction":
        def raise_invalid(message: str, original_error: Exception = None):
            error_message = f"Invalid PyAutoGUI action string: {message}"
            if original_error:
                raise ValueError(error_message) from original_error
            else:
                raise ValueError(error_message)

        try:
            tree = ast.parse(action_string)
            call = tree.body[0].value

            if (
                not isinstance(call, ast.Call)
                or not isinstance(call.func, ast.Attribute)
                or call.func.value.id != "pyautogui"
            ):
                raise_invalid("Invalid syntax")

            action_type = GUIActionType(call.func.attr)

            # Use the new infer_kwargs method
            args = cls.infer_kwargs(action_string)

            if len(args) == 1 and "args" in args:
                args = args["args"]

            return cls(action_type=action_type, args=args)
        except Exception as e:
            raise_invalid(action_string, e)

    @staticmethod
    def _parse_ast_node(node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.List):
            return [PyAutoGUIAction._parse_ast_node(elt) for elt in node.elts]
        elif isinstance(node, ast.Tuple):
            return tuple(PyAutoGUIAction._parse_ast_node(elt) for elt in node.elts)
        elif isinstance(node, ast.Dict):
            return {
                PyAutoGUIAction._parse_ast_node(key): PyAutoGUIAction._parse_ast_node(value)
                for key, value in zip(node.keys, node.values)
            }
        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -PyAutoGUIAction._parse_ast_node(node.operand)
            elif isinstance(node.op, ast.UAdd):
                return PyAutoGUIAction._parse_ast_node(node.operand)
            else:
                raise TypeError(f"Unsupported unary operator: {type(node.op)}")
        else:
            raise TypeError(f"Unsupported AST node type: {type(node)}")

    @staticmethod
    def infer_kwargs(func_string: str) -> Dict[str, Any]:
        # Parse the function call string
        tree = ast.parse(func_string)
        call = tree.body[0].value

        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Attribute)
            or call.func.value.id != "pyautogui"
        ):
            raise ValueError("Invalid PyAutoGUI function string")

        func_name = call.func.attr
        args = [PyAutoGUIAction._parse_ast_node(arg) for arg in call.args]
        kwargs = {kw.arg: PyAutoGUIAction._parse_ast_node(kw.value) for kw in call.keywords}

        # Get the function from pyautogui module
        func = getattr(pyautogui, func_name)

        # Get the signature of the function
        sig = inspect.signature(func)

        # Create a dictionary to store the keyword arguments
        result = {}

        # Match the provided arguments to the function parameters
        params = list(sig.parameters.values())
        for i, value in enumerate(args):
            if i < len(params):
                param = params[i]
                if param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD:
                    result[param.name] = value
                elif param.kind == inspect.Parameter.VAR_POSITIONAL:
                    # If we encounter *args, store the remaining args as a list
                    result[param.name] = args[i:]
                    break

        # Add any explicitly provided keyword arguments
        result.update(kwargs)

        return result

    def normalize_args(self, image_size: Tuple[int, int]) -> "PyAutoGUIAction":
        """Normalize the x y coordinates in the arguments to be between 0 and 1."""
        if self.action_type in [GUIActionType.CLICK, GUIActionType.DOUBLE_CLICK, GUIActionType.RIGHT_CLICK]:
            x, y = self.args["x"], self.args["y"]
            img_width, img_height = image_size
            self.args["x"] = x / img_width
            self.args["y"] = y / img_height


class ComputerActionType(str, Enum):
    TRIPLE_CLICK = "tripleClick"
    WAIT = "wait"
    TERMINATE = "terminate"


triple_click_func = {
    "name": "computer.triple_click",
    "description": "Triple click on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the triple click",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the triple click",
            },
        },
        "required": ["x", "y"],
    },
}

wait_func = {
    "name": "computer.wait",
    "description": "Wait for a specified duration",
    "parameters": {
        "type": "object",
        "properties": {
            "duration": {
                "type": "number",
                "description": "The duration to wait in seconds",
            },
        },
        "required": ["duration"],
    },
}

terminate_func = {
    "name": "computer.terminate",
    "description": "Terminate the current task and report its completion status",
    "parameters": {
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "enum": ["success", "failure"],
                "description": "The status of the task",
            },
        },
        "required": ["status"],
    },
}


class ComputerAction(BaseModel):
    action_type: ComputerActionType
    target: Optional[GUIElement] = Field(None, description="The target element") 
    args: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, description="Arguments for the action")

    def to_command(self) -> str:
        def convert_and_round(value):
            if isinstance(value, float):
                return round(value, 4)  # Round to 4 decimal places
            if isinstance(value, list):
                return [convert_and_round(v) for v in value]
            return value

        if isinstance(self.args, list):
            args_str = ", ".join(repr(convert_and_round(v)) for v in self.args)
        elif len(self.args) == 1 and next(iter(self.args.keys())) in ["clicks", "amount"]:
            # For single-argument functions like scroll, just use the value
            args_str = repr(convert_and_round(next(iter(self.args.values()))))
        else:
            args_str = ", ".join(f"{k}={repr(convert_and_round(v))}" for k, v in self.args.items())
        return f"computer.{self.action_type.value}({args_str})"

    @field_validator("args")
    @classmethod
    def validate_args(cls, v):
        if v is not None:
            if "x" in v and "y" in v:
                x, y = v["x"], v["y"]
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("x and y coordinates must be between 0 and 1")
        return v

    @classmethod
    def from_string(cls, action_string: str) -> "ComputerAction":
        # Extract action type and arguments from the string
        parts = action_string.split("(", 1)
        action_type = parts[0].split(".")[-1]
        args_str = parts[1].rstrip(")")

        # Parse arguments
        args = {}
        if args_str:
            for arg in args_str.split(","):
                key, value = arg.split("=")
                key = key.strip()
                value = value.strip()

                # Convert value to appropriate type
                if value.startswith("'") or value.startswith('"'):
                    args[key] = value.strip("'\"")
                elif value.lower() == "true":
                    args[key] = True
                elif value.lower() == "false":
                    args[key] = False
                elif "." in value:
                    args[key] = float(value)
                else:
                    args[key] = int(value)

        # Create and return ComputerAction instance
        return cls(action_type=ComputerActionType(action_type), args=args)


class BrowserActionType(str, Enum):
    SELECT_OPTION = "select_option"
    CLEAR = "clear"


select_option_func = {
    "name": "browser.select_option",
    "description": "Select an option from a dropdown menu",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the dropdown menu",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the dropdown menu",
            },
            "value": {
                "type": "string",
                "description": "The value of the option to select",
            },
        },
        "required": ["x", "y", "value"],
    },
}

clear_func = {
    "name": "browser.clear",
    "description": "Clear the text from an input field",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the input field",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the input field",
            },
        },
        "required": ["x", "y"],
    },
}


class BrowserAction(BaseModel):
    action_type: BrowserActionType
    target: Optional[Any] = Field(None, description="The target element")
    args: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, description="Arguments for the action")

    def to_command(self) -> str:
        def convert_and_round(value):
            if isinstance(value, float):
                return round(value, 4)  # Round to 4 decimal places
            return value

        if isinstance(self.args, list):
            args_str = ", ".join(repr(convert_and_round(v)) for v in self.args)
        elif len(self.args) == 1 and next(iter(self.args.keys())) in ["clicks", "amount"]:
            # For single-argument functions like scroll, just use the value
            args_str = repr(convert_and_round(next(iter(self.args.values()))))
        else:
            args_str = ", ".join(f"{k}={repr(convert_and_round(v))}" for k, v in self.args.items())
        return f"browser.{self.action_type.value}({args_str})"


class MobileActionType(str, Enum):
    Swipe = "swipe"
    Home = "home"
    Back = "back"
    Wait = "wait"
    LongPress = "long_press"
    OpenApp = "open_app"
    # Terminate = "terminate"


swipe_func = {
    "name": "mobile.swipe",
    "description": "Swipe on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "from_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The starting coordinates of the swipe",
            },
            "to_coord": {
                "type": "array",
                "items": {"type": "number"},
                "description": "The ending coordinates of the swipe",
            },
        },
        "required": ["from_coord", "to_coord"],
    },
}

home_func = {"name": "mobile.home", "description": "Press the home button"}

back_func = {"name": "mobile.back", "description": "Press the back button"}

wait_func = {
    "name": "mobile.wait",
    "description": "wait for the change to happen",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {
                "type": "number",
                "description": "The seconds to wait",
            },
        },
        "required": ["seconds"],
    },
}

long_press_func = {
    "name": "mobile.long_press",
    "description": "Long press on the screen",
    "parameters": {
        "type": "object",
        "properties": {
            "x": {
                "type": "number",
                "description": "The x coordinate of the long press",
            },
            "y": {
                "type": "number",
                "description": "The y coordinate of the long press",
            },
        },
        "required": ["x", "y"],
    },
}

open_app_func = {
    "name": "mobile.open_app",
    "description": "Open an app on the device",
    "parameters": {
        "type": "object",
        "properties": {
            "app_name": {
                "type": "string",
                "description": "The name of the app to open",
            },
        },
        "required": ["app_name"],
    },
}

# termi_func ={
#     "name": "mobile.terminate",
#     "description": "Terminate the current task and report its completion status",
#     "parameters": {
#         "type": "object",
#         "properties": {
#             "status": {
#                 "type": "string",
#                 "enum": ["success"],
#                 "description": "The status of the task",
#             },
#         },
#         "required": ["status"],
#     },
# }


class MobileAction(BaseModel):
    action_type: MobileActionType
    target: Optional[GUIElement] = Field(None, description="The target element")
    args: Union[Dict[str, Any], List[Any]] = Field(default_factory=dict, description="Arguments for the action")

    def to_command(self) -> str:
        def convert_and_round(value):
            if isinstance(value, float):
                return round(value, 4)  # Round to 4 decimal places
            if isinstance(value, list):
                return [convert_and_round(v) for v in value]
            return value

        if isinstance(self.args, list):
            args_str = ", ".join(repr(convert_and_round(v)) for v in self.args)
        elif len(self.args) == 1 and next(iter(self.args.keys())) in ["clicks", "amount"]:
            # For single-argument functions like scroll, just use the value
            args_str = repr(convert_and_round(next(iter(self.args.values()))))
        else:
            args_str = ", ".join(f"{k}={repr(convert_and_round(v))}" for k, v in self.args.items())
        return f"mobile.{self.action_type.value}({args_str})"

    @field_validator("args")
    @classmethod
    def validate_args(cls, v):
        if v is not None:
            if "x" in v and "y" in v:
                x, y = v["x"], v["y"]
                if not (0 <= x <= 1 and 0 <= y <= 1):
                    raise ValueError("x and y coordinates must be between 0 and 1")
            elif "from_coord" in v and "to_coord" in v:
                from_coord, to_coord = v["from_coord"], v["to_coord"]
                if not (
                    0 <= from_coord[0] <= 1
                    and 0 <= from_coord[1] <= 1
                    and 0 <= to_coord[0] <= 1
                    and 0 <= to_coord[1] <= 1
                ):
                    raise ValueError("from_coord and to_coord coordinates must be between 0 and 1")
        return v

    @classmethod
    def from_string(cls, action_string: str) -> "MobileAction":
        # Extract action type and arguments from the string
        parts = action_string.split("(", 1)
        action_type = parts[0].split(".")[-1]
        args_str = parts[1].rstrip(")")

        # Parse arguments
        args = {}
        if args_str:
            for arg in args_str.split(","):
                key, value = arg.split("=")
                key = key.strip()
                value = value.strip()

                # Convert value to appropriate type
                if value.startswith("'") or value.startswith('"'):
                    args[key] = value.strip("'\"")
                elif value.lower() == "true":
                    args[key] = True
                elif value.lower() == "false":
                    args[key] = False
                elif "." in value:
                    args[key] = float(value)
                else:
                    args[key] = int(value)

        # Create and return MobileAction instance
        return cls(action_type=MobileActionType(action_type), args=args)


answer_func = {
    "name": "answer",
    "description": "Answer a question",
    "parameters": {
        "type": "object",
        "properties": {
            "answer": {
                "type": "string",
                "description": "The answer to the question",
            },
        },
        "required": ["answer"],
    },
}


class CommunicationActionType(str, Enum):
    ANSWER = "answer"


class CommunicationAction(BaseModel):
    action_type: CommunicationActionType
    target: Optional[GUIElement] = Field(None, description="The target element")
    args: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the action")

    @model_validator(mode="after")
    def validate_args(self):
        if self.action_type == CommunicationActionType.ANSWER and "answer" not in self.args:
            raise ValueError("The 'answer' argument is required for the ANSWER action type")
        return self

    def to_command(self) -> str:
        if self.action_type == CommunicationActionType.ANSWER:
            return f"answer('{self.args['answer']}')"
        else:
            raise ValueError(f"Unsupported action type: {self.action_type}")

    @classmethod
    def from_string(cls, action_string: str) -> "CommunicationAction":
        parts = action_string.split("(", 1)
        action_type = parts[0]
        args_str = parts[1].rstrip(")")

        args = {}
        if args_str:
            key, value = args_str.split("=")
            args[key.strip()] = value.strip().strip("'\"")

        return cls(action_type=CommunicationActionType(action_type), args=args)


class GUIAction(Action):
    observation: Optional[str] = Field(None, description="The observation before the action")
    thought: Optional[str] = Field(None, description="The thought before the action")
    instruction: str = Field(..., description="The description/thought provided for the action")
    guiactions: list[PyAutoGUIAction | BrowserAction | MobileAction | CommunicationAction | ComputerAction] = Field(
        ..., description="The GUI actions to perform"
    )
