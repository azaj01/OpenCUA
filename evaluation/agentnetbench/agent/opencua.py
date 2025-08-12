#!/usr/bin/env python3
"""
OpenCUA agent implementation for the AgentNetBench benchmark.

This agent mirrors the prompting, parsing and action extraction logic from
`draft/opencua_eval_all_in_one.py`, adapted to the `BaseAgent` interface so it
can be run inside the benchmark framework.

Key features:
- Prompt styles: L1/L2/L3 and their short variants
- History control: include previous steps with text and optional images
- Parsing of pyautogui/computer.* lines from model outputs
- Extraction of structured actions for evaluation
"""

from __future__ import annotations

import base64
import re
import random
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .base_agent import BaseAgent


# -------- Prompt templates (mirrored from opencua_eval_all_in_one.py) --------
STEP_TEMPLATE = "# Step {step_num}:\n"
INSTRUTION_TEMPLATE = (
    "\n# Task Instruction:\n{instruction}\n\n"
    "Please generate the next move according to the screenshot, task instruction and previous steps (if provided).\n"
)

L3_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nObservation:\n  - Describe the current computer state based on the full screenshot in detail. \n  - Application Context:\n    - The active application\n    - The active window or page\n    - Overall layout and visible interface\n  - Key Elements:\n    - Menu items and toolbars \n    - Buttons and controls\n    - Text fields and content\n    - Dialog boxes or popups\n    - Error messages or notifications\n    - Loading states\n    - Other key elements\n  - Describe any content, elements, options, information or clues that are possibly relevant to achieving the task goal, including their name, content, or shape (if possible).\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n  - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}\n',
]

L2_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n  - For Text Input Actions:\n    - Note current cursor position\n    - Consolidate repetitive actions (specify count for multiple keypresses)\n    - Describe expected final text outcome\n    - Use first-person perspective in reasoning\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

L1_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nAction:\n  Provide clear, concise, and actionable instructions:\n  - If the action involves interacting with a specific target:\n    - Describe target explicitly without using coordinates\n    - Specify element names when possible (use original language if non-English)\n    - Describe features (shape, color, position) if name unavailable\n    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")\n  - if the action involves keyboard actions like \'press\', \'write\', \'hotkey\':\n    - Consolidate repetitive keypresses with count\n    - Specify expected text outcome for typing actions\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

L3_SHORT_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nObservation: Describe the current computer state based on the full screenshot in detail. Provide any information that is possibly relevant to achieving the task goal and any elements that may affect the task execution, such as pop-ups, notifications, error messages, loading states, etc..\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n\nAction:\n  Provide clear, concise, and actionable instructions.\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

L2_SHORT_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nThought:\n  - Step by Step Progress Assessment:\n    - Analyze completed task parts and their contribution to the overall goal\n    - Reflect on potential errors, unexpected results, or obstacles\n    - If previous action was incorrect, predict a logical recovery step\n  - Next Action Analysis:\n    - List possible next actions based on current state\n    - Evaluate options considering current state and previous actions\n    - Propose most logical next action\n    - Anticipate consequences of the proposed action\n\nAction:\n  Provide clear, concise, and actionable instructions.\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

L1_SHORT_SYSTEM_PROMPTS = [
    'You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.\n\nFor each step, provide your response in this format:\n\nAction:\n  Provide clear, concise, and actionable instructions.\n\nFinally, output the action as PyAutoGUI code or the following functions:\n- {"name": "computer.triple_click", "description": "Triple click on the screen", "parameters": {"type": "object", "properties": {"x": {"type": "number", "description": "The x coordinate of the triple click"}, "y": {"type": "number", "description": "The y coordinate of the triple click"}}, "required": ["x", "y"]}}\n- {"name": "computer.terminate", "description": "Terminate the current task and report its completion status", "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["success", "failure"], "description": "The status of the task"}}, "required": ["status"]}}'
]

ACTION_HISTORY_TEMPLATE = "## Action:\n{action}\n"
THOUGHT_HISTORY_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n"
OBSERVATION_HISTORY_TEMPLATE = (
    "## Observation:\n{observation}\n\n## Thought:\n{thought}\n\n## Action:\n{action}\n"
)

L1_RESPONSE_TEMPLATE = "## Action:\n{action}\n\n## Code:\n{code}\n"
L2_RESPONSE_TEMPLATE = "## Thought:\n{thought}\n\n## Action:\n{action}\n\n## Code:\n{code}\n"
L3_RESPONSE_TEMPLATE = (
    "## Observation:\n{observation}\n\n## Thought:\n{thought}\n\n## Action:\n{action}\n\n## Code:\n{code}\n"
)


class OpenCUA(BaseAgent):
    """OpenCUA agent aligned with the BaseAgent interface."""

    def __init__(
        self,
        model: str,
        client: Any,
        *,
        l_number: str = "l2",  # one of: l1, l2, l3, l1_short, l2_short, l3_short
        history: str = "thought",  # one of: action, thought, observation
        image: str = "image_3",  # one of: image_1, image_3, image_5
        max_history_length: int = 10,
        max_detail_length: int = 0,
    ) -> None:
        super().__init__(model, client)
        self.l_number = l_number
        self.history = history
        self.image = image
        self.max_history_length = max_history_length
        self.max_detail_length = max_detail_length

        # Choose system and response templates
        if l_number == "l1":
            self.SYSTEM_PROMPTS = random.choice(L1_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L1_RESPONSE_TEMPLATE
        elif l_number == "l2":
            self.SYSTEM_PROMPTS = random.choice(L2_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L2_RESPONSE_TEMPLATE
        elif l_number == "l3":
            self.SYSTEM_PROMPTS = random.choice(L3_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L3_RESPONSE_TEMPLATE
        elif l_number == "l1_short":
            self.SYSTEM_PROMPTS = random.choice(L1_SHORT_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L1_RESPONSE_TEMPLATE
        elif l_number == "l2_short":
            self.SYSTEM_PROMPTS = random.choice(L2_SHORT_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L2_RESPONSE_TEMPLATE
        elif l_number == "l3_short":
            self.SYSTEM_PROMPTS = random.choice(L3_SHORT_SYSTEM_PROMPTS)
            self.RESPONSE_TEMPLATE = L3_RESPONSE_TEMPLATE
        else:
            raise ValueError(f"Unsupported l_number: {l_number}")

        # Choose history template
        if history == "action":
            self.HISTORY_TEMPLATE = ACTION_HISTORY_TEMPLATE
        elif history == "thought":
            self.HISTORY_TEMPLATE = THOUGHT_HISTORY_TEMPLATE
        elif history == "observation":
            self.HISTORY_TEMPLATE = OBSERVATION_HISTORY_TEMPLATE
        else:
            raise ValueError(f"Unsupported history type: {history}")

        # Choose number of images to include
        if image == "image_1":
            self.image_num = 1
        elif image == "image_3":
            self.image_num = 3
        elif image == "image_5":
            self.image_num = 5
        else:
            raise ValueError(f"Unsupported image setting: {image}")

    # ----------------------------- Utilities -----------------------------
    @staticmethod
    def _encode_image_bytes(image_bytes: bytes) -> str:
        return base64.b64encode(image_bytes).decode("utf-8")

    # ----------------------------- Prompting -----------------------------
    def prompt(self, trajectory: Dict[str, Any], current_step: int) -> List[Dict[str, Any]]:
        """Generate messages following the all-in-one evaluator's formatting."""
        messages: List[Dict[str, Any]] = []
        messages.append({"role": "system", "content": self.SYSTEM_PROMPTS})

        # Gather previous steps' data
        steps: List[Dict[str, Any]] = trajectory.get("steps", [])
        total_steps = len(steps)
        if current_step < 0 or current_step >= total_steps:
            return messages

        # Backfill previous images to include
        prev_indices: List[int] = list(range(0, current_step))
        previous_images_b64: List[str] = []

        # Only include up to image_num - 1 previous images (as in the all-in-one logic)
        # Prefer the most recent steps
        image_slots = min(self.image_num - 1, len(prev_indices))
        indices_with_images = prev_indices[-image_slots:] if image_slots > 0 else []

        for idx in indices_with_images:
            prev_step = steps[idx]
            image_file = prev_step.get("image")
            if image_file and self.image_dir:
                image_bytes = self.load_image(image_file, self.image_dir)
                previous_images_b64.append(self._encode_image_bytes(image_bytes))

        # Build history messages
        # We iterate all previous steps, adding images for the last (image_num-1) ones
        for i, idx in enumerate(prev_indices):
            step = steps[idx]
            inner = step.get("inner_monologue", {})

            # Determine if this previous step should include an image
            include_image = False
            if indices_with_images:
                include_image = idx in indices_with_images

            if include_image:
                # Attach image as user
                # Map to the corresponding image in previous_images_b64
                b64_idx = indices_with_images.index(idx)
                if b64_idx < len(previous_images_b64):
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "detail": "auto",
                                        "url": f"data:image/jpeg;base64,{previous_images_b64[b64_idx]}",
                                    },
                                }
                            ],
                        }
                    )

            # Add either detailed or brief history template
            # Respect max_history_length and max_detail_length
            if idx >= max(0, current_step - self.max_history_length):
                if idx >= max(0, current_step - self.max_detail_length):
                    # Use detailed RESPONSE_TEMPLATE
                    messages.append(
                        {
                            "role": "assistant",
                            "content": STEP_TEMPLATE.format(step_num=idx + 1)
                            + self.RESPONSE_TEMPLATE.format(
                                observation=inner.get("observation", ""),
                                thought=inner.get("thought", ""),
                                action=inner.get("low_level_instruction", ""),
                                code=step.get("action", ""),
                            ),
                        }
                    )
                else:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": STEP_TEMPLATE.format(step_num=idx + 1)
                            + self.HISTORY_TEMPLATE.format(
                                observation=inner.get("observation", ""),
                                thought=inner.get("thought", ""),
                                action=inner.get("low_level_instruction", ""),
                            ),
                        }
                    )

        # Current step: add image and instruction
        cur_step = steps[current_step]
        cur_image_file = cur_step.get("image")
        current_image_b64: Optional[str] = None
        if cur_image_file and self.image_dir:
            img_bytes = self.load_image(cur_image_file, self.image_dir)
            current_image_b64 = self._encode_image_bytes(img_bytes)

        user_content: List[Dict[str, Any]] = []
        if current_image_b64:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"detail": "auto", "url": f"data:image/jpeg;base64,{current_image_b64}"},
                }
            )
        # Instruction text
        instruction_text = trajectory.get("high_level_task_description") or trajectory.get("instruction") or ""
        user_content.append({"type": "text", "text": INSTRUTION_TEMPLATE.format(instruction=instruction_text)})
        messages.append({"role": "user", "content": user_content})

        return messages

    # ----------------------------- Parsing ------------------------------
    def parse_response(self, response: str, trajectory: Optional[Dict[str, Any]] = None, step_idx: Optional[int] = None) -> Optional[str]:
        """Parse model output, extracting pyautogui/computer lines."""
        if response is None:
            return None

        lines = response.split("\n")
        action_lines: List[str] = []

        # First pass: lines that start with commands
        for raw in lines:
            line = raw.strip()
            if line.startswith("pyautogui.") or line.startswith("computer."):
                action_lines.append(line)

        if action_lines:
            return "\n".join(action_lines)

        # Second pass: find commands anywhere within lines
        for raw in lines:
            line = raw.strip()
            if "pyautogui." in line:
                parts = line.split("pyautogui.")
                action_lines.append("pyautogui." + parts[1].strip())
            elif "computer." in line:
                parts = line.split("computer.")
                action_lines.append("computer." + parts[1].strip())

        return "\n".join(action_lines) if action_lines else None

    # ----------------------- Action extraction --------------------------
    def extract_actions(self, action: str) -> List[Tuple[str, Any]]:
        """Extract (type, value) tuples from parsed action string.

        Follows the logic of extract_actions() in opencua_eval_all_in_one.py.
        """
        if not action:
            return []

        actions: List[Tuple[str, Any]] = []

        action_lines = action.strip().split("\n")
        for raw in action_lines:
            line = raw.strip()

            # computer.terminate
            if line.startswith("computer.terminate"):
                status_match = re.search(r"status=['\"](\w+)['\"]", line)
                if status_match:
                    actions.append(("terminate", status_match.group(1)))
                    continue

            # computer.triple_click
            if line.startswith("computer.triple_click"):
                coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
                if coord_match:
                    x, y = map(float, coord_match.groups())
                    actions.append(("triple_click", (x, y)))
                    continue

            # pyautogui.*
            if line.startswith("pyautogui."):
                coord_match = re.search(r"x=([\d.]+),\s*y=([\d.]+)", line)
                if coord_match:
                    x, y = map(float, coord_match.groups())
                    if "click" in line and "doubleClick" not in line and "rightClick" not in line:
                        actions.append(("click", (x, y)))
                    elif "moveTo" in line:
                        actions.append(("moveTo", (x, y)))
                    elif "doubleClick" in line:
                        actions.append(("doubleClick", (x, y)))
                    elif "rightClick" in line:
                        actions.append(("rightClick", (x, y)))
                    elif "dragTo" in line:
                        actions.append(("dragTo", (x, y)))

                # write(message=...)
                write_match = re.search(r"message=['\"](.+?)['\"]", line)
                if write_match:
                    text = write_match.group(1)
                    actions.append(("write", text))

                # press/hotkey keys=[...]
                keys_match = re.findall(r"keys=\[(.*?)\]", line)
                if keys_match:
                    key_string = keys_match[0]
                    key_list = re.findall(r"['\"]([^'\"]*)['\"]|(\w+)", key_string)
                    keys = [m[0] or m[1] for m in key_list if m[0] or m[1]]
                    normalized_keys: List[str] = []
                    for k in keys:
                        k = k.strip()
                        normalized_keys.append("ctrl" if k.lower() in ("cmd", "command") else k)
                    if "hotkey" in line:
                        actions.append(("hotkey", normalized_keys))
                    else:
                        actions.append(("press", normalized_keys))

                # scroll
                scroll_match = re.search(r"pyautogui\.scroll\(([-\d]+)\)", line)
                if scroll_match:
                    actions.append(("scroll", int(scroll_match.group(1))))

        return actions


