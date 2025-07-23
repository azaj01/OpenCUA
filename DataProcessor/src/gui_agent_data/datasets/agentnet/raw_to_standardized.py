import json
import re

from gui_agent_data.schema.action.gui import (
    ComputerAction,
    GUIAction,
    PyAutoGUIAction,
    triple_click_func,
    terminate_func
)
from gui_agent_data.schema.observation.image import ImageObservation
from gui_agent_data.schema.observation.text import TextObservation
from gui_agent_data.schema.trajectory import Trajectory
from tqdm import tqdm

# system_instruction = f"""You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.
# For each step, provide your response in this format:\n\n
# Observation: Describe the current screen state, including:\n
# - The active application and visible window elements\n
# - Analysis of previous action results if any\n
# - Whether previous actions completed successfully\n
# - Any loading states or unresponsive elements\n\n
# Thought: Analyze the situation, including:\n
# - Progress toward the overall goal\n
# - Reasoning about possible next actions\n
# - Potential obstacles or errors\n
# - Expected outcomes of the next action\n\n
# Action: Provide clear instructions that:\n
# - Describe the exact target without coordinates\n
# - Match the ground truth action sequence\n
# - Specify key presses or text input clearly\n\n
# Finally, output the corresponding PyAutoGUI code for the action

# You have access to the following functions:
# - {json.dumps(triple_click_func)}
# - {json.dumps(terminate_func)}
# """

system_instruction = f'''You are a GUI agent. You are given a task and a screenshot of the screen. You need to perform a series of pyautogui actions to complete the task.

For each step, provide your response in this format:

Observation:
  - Describe the current computer state based on the full screenshot in detail. 
  - Application Context:
    - The active application
    - The active window or page
    - Overall layout and visible interface
  - Key Elements:
    - Menu items and toolbars 
    - Buttons and controls
    - Text fields and content
    - Dialog boxes or popups
    - Error messages or notifications
    - Loading states
    - Other key elements
  - Describe any content, elements, options, information or clues that are possibly relevant to achieving the task goal, including their name, content, or shape (if possible).

Thought:
  - Step by Step Progress Assessment:
    - Analyze completed task parts and their contribution to the overall goal
    - Reflect on potential errors, unexpected results, or obstacles
    - If previous action was incorrect, predict a logical recovery step
  - Next Action Analysis:
    - List possible next actions based on current state
    - Evaluate options considering current state and previous actions
    - Propose most logical next action
    - Anticipate consequences of the proposed action
  - For Text Input Actions:
    - Note current cursor position
    - Consolidate repetitive actions (specify count for multiple keypresses)
    - Describe expected final text outcome
  - Use first-person perspective in reasoning

Action:
  Provide clear, concise, and actionable instructions:
  - If the action involves interacting with a specific target:
    - Describe target explicitly without using coordinates
    - Specify element names when possible (use original language if non-English)
    - Describe features (shape, color, position) if name unavailable
    - For window control buttons, identify correctly (minimize "—", maximize "□", close "X")
  - if the action involves keyboard actions like 'press', 'write', 'hotkey':
    - Consolidate repetitive keypresses with count
    - Specify expected text outcome for typing actions

Finally, output the action as PyAutoGUI code or the following functions:
- {json.dumps(triple_click_func)}
- {json.dumps(terminate_func)}
'''

SCROLL_DIRECTION_MAP = {
    "\u2b07\ufe0f": "down",  # ⬇️
    "↙": "down-left",  # ↙️
    "\u2b06\ufe0f": "up",  # ⬆️
    "↘": "down-right",  # ↘️
    "↖": "up-left",  # ↖
    "↗️": "up-right",  # ↗️
    "\u2b05\ufe0f": "left",  # ⬅️
    "\u27a1\ufe0f": "right",  # ➡️
}

SPECIAL_SYMBOLS = set(["-", "_", "+", "~", "!", "@", "#", "$", "%", "^", "&", "*", 
                    "(", ")", "[", "]", "{", "}", "|", ":", '"', "'", "<", ">", 
                    "?", "/", "=", "–", ".", ",", ";", "'", "`", ".", "=", "\\",
                    "）", "（"])


def preprocess_events(events):
    for index, item in enumerate(events):
        # if item["action"] == "press" and "$cmd$" in item["description"]:
        #     item["description"] = item["description"].replace("$cmd$", "$ctrl$")
        if index + 1 >= len(events):
            continue
        # restrict that only "Type: c" in the description
        if (
            item["action"] == "mouse_press"
            and events[index + 1]["action"] == "type"
            and ( events[index + 1]["description"] == "Type: c" or events[index+1]["description"] == "Type: C" )
        ):
            # change two actions into drag and press (ctrl + c)
            events[index]["action"] = "drag"
            events[index]["description"] = "Drag from (0,0) to (0, 0)"
            events[index + 1]["action"] = "press"
            events[index + 1]["description"] = "Press: $cmd$ + c"
        elif (
            item["action"] == "drag"
            and events[index + 1]["action"] == "type"
            and ("Type: c" in events[index + 1]["description"] or "Type: v" in events[index + 1]["description"])
        ):
            # print("here")
            events[index + 1]["action"] = "press"
            if "Type: c" in events[index + 1]["description"]:
                events[index + 1]["description"] = "Press: $cmd$ + c"
            elif "Type: v" in events[index + 1]["description"]:
                events[index + 1]["description"] = "Press: $cmd$ + v"

    return events


def parse_scroll_to_cardinal(scroll_text):

    directions = {"up": 0, "down": 0, "left": 0, "right": 0}

    # Remove the "Scroll" prefix and split into parts
    actions = scroll_text.replace("Scroll ", "").split()


    for action in actions:
        if "\u00d7" in action:  # × multiplication sign
            direction, magnitude = action.split("\u00d7")
            print(f"direction: {repr(direction)}, magnitude: {magnitude}")
            magnitude = int(magnitude)

            # Get direction from pre-defined map
            if direction in SCROLL_DIRECTION_MAP:
                mapped_direction = SCROLL_DIRECTION_MAP[direction]
                print(f"mapped_direction: {mapped_direction}")

                # Handle cardinal directions
                if mapped_direction in directions:
                    directions[mapped_direction] += magnitude
                # Handle diagonal directions
                elif mapped_direction == "down-left":
                    directions["down"] += magnitude
                    directions["left"] += magnitude
                elif mapped_direction == "down-right":
                    directions["down"] += magnitude
                    directions["right"] += magnitude
                elif mapped_direction == "up-left":
                    directions["up"] += magnitude
                    directions["left"] += magnitude
                elif mapped_direction == "up-right":
                    directions["up"] += magnitude
                    directions["right"] += magnitude

    return directions

def record_scroll_to_trace(scroll_text,trace):
    # record = {
    #     "coordinates": {'x': 0, 'y': 0},
    #     "directions": {"up": 0, "down": 0, "left": 0, "right": 0}
    # }
    directions = {"up": 0, "down": 0, "left": 0, "right": 0}

    # Remove the "Scroll" prefix and split into parts
    actions = scroll_text.replace("Scroll ", "").split()
    accum_magnitude = 0
    for action in actions:
        if "\u00d7" in action:  # × multiplication sign
            direction, magnitude = action.split("\u00d7")
            index = accum_magnitude
            x, y = trace[index]["x"], trace[index]["y"]
            accum_magnitude += int(magnitude)


def get_action_type(action):
    action = action.lower()
    if "single right click" in action:
        return "rightClick"
    elif "double right click" in action:
        return "rightClick"
    elif "single left click" in action:
        return "click"
    elif "single x1 click" in action:
        return "click"
    elif "double left click" in action:
        return "doubleClick"
    elif "single middle click" in action:
        return "middleClick"
    elif "triple left click" in action:
        return "tripleClick"
    elif "mouse long press left button" in action:
        return "click"
    elif "mouse long press right button" in action:
        return "rightClick"
    elif "mouse long press middle button" in action:
        return "middleClick"
    elif "type" in action:
        return "write"
    elif "press" in action:
        return "press"
    elif "drag" in action:
        return "dragTo"
    elif "scroll" in action:
        return "scroll"
    elif "terminate" in action:
        return "terminate"
    else:
        raise ValueError(f"Unknown action: {action}")


def reduce_actions(actionlist):
    reduced_actionlist = []
    unknown_flag = False
    for action in actionlist:
        if (
            action.action_type == "press"
        ):
            unknown_flag = False
            for key in action.args["keys"]:
                if "Unknown" in key:
                    unknown_flag = True
                    break
            if unknown_flag:
                continue
        

        if (
            action.action_type == "press"
            and len(reduced_actionlist) >= 1
            and reduced_actionlist[-1].action_type == "write"
            and action.args["keys"][0] == "space"
            and len(action.args["keys"]) == 1
        ):
            reduced_actionlist[-1].args["message"] += " "
        elif (
            action.action_type == "write"
            and len(reduced_actionlist) >= 1
            and reduced_actionlist[-1].action_type == "write"
        ):
            reduced_actionlist[-1].args["message"] += action.args["message"]
        elif (
            action.action_type == "press"
            and len(reduced_actionlist) >= 1
            and reduced_actionlist[-1].action_type == "write"
            and action.args["keys"][0] == "backspace"
            and len(action.args["keys"]) == 1
        ):
            reduced_actionlist[-1].args["message"] = reduced_actionlist[-1].args["message"][:-1]
        else:
            reduced_actionlist.append(action)
            
    return reduced_actionlist


def reduce_content(episode_id, step_num, content):
    # print(type(content))
    # print(len(content))
    reduced_content = []
    continue_flag = False
    # print("haha")
    for index, item in enumerate(content):
        if continue_flag:
            continue_flag = False
            continue
        # print(index)
        try:
            if isinstance(item, (TextObservation, ImageObservation)):
                # print("here")
                reduced_content.append(item)
            # elif "guiactions" in item and item["guiactions"][0]["action_type"] == "write" and "guiactions" in reduced_content[-2] and reduced_content[-2]["guiactions"][-1]["action_type"] == "write":
            elif (
                isinstance(item, GUIAction)
                and len(item.guiactions) == 0
            ):
                if isinstance(reduced_content[-1], ImageObservation):
                    reduced_content.pop()
                elif isinstance(reduced_content[-1], TextObservation):
                    task_instruction = reduced_content.pop()
                    reduced_content.pop()
                    reduced_content.append(content[index+1])
                    reduced_content.append(task_instruction)
                    continue_flag = True
                continue
            elif (
                isinstance(item, GUIAction)
                and len(item.guiactions) > 0
                and item.guiactions[0].action_type == "write"
                and isinstance(reduced_content[-2], GUIAction)
                and reduced_content[-2].guiactions[-1].action_type == "write"
            ):
                # print("here1")
                # pop the last two items and save the second last item
                reduced_content.pop()
                second_last_item = reduced_content.pop()
                # append the new guiaction list to the last item
                second_last_item.instruction = second_last_item.instruction + " " + item.instruction
                second_last_item.guiactions.extend(item.guiactions)  # Changed from dict to attribute access
                # reduce the action list of second last item
                second_last_item.guiactions = reduce_actions(second_last_item.guiactions)
                # append the last item back
                reduced_content.append(second_last_item)
            elif (
                isinstance(item, GUIAction)
                and item.guiactions[0].action_type == "click"
                and isinstance(reduced_content[-2], GUIAction)
                and reduced_content[-2].guiactions[-1].action_type == "click"
                and item.guiactions[0].args["x"] == reduced_content[-2].guiactions[-1].args["x"]
                and item.guiactions[0].args["y"] == reduced_content[-2].guiactions[-1].args["y"]
            ):
                # pop the last two items and save the second last item
                reduced_content.pop()
                second_last_item = reduced_content.pop()
                second_last_item.instruction = second_last_item.instruction + " " + item.instruction
                # turn the click into double click
                second_last_item.guiactions[-1].action_type = "doubleClick"
                # second_last_item.guiactions.extend(item.guiactions)
                reduced_content.append(second_last_item)
            elif (
                isinstance(item, GUIAction)
                and item.guiactions[0].action_type == "hotkey"
                and len(item.guiactions[0].args["keys"]) == 1
                and (item.guiactions[0].args["keys"][0] == "shift" \
                    or item.guiactions[0].args["keys"][0] == "ctrl" \
                    or item.guiactions[0].args["keys"][0] == "cmd")
            ):
                # pop the last item
                if isinstance(reduced_content[-1], ImageObservation):
                    reduced_content.pop()
                elif isinstance(reduced_content[-1], TextObservation):
                    task_instruction = reduced_content.pop()
                    reduced_content.pop()
                    reduced_content.append(content[index+1])
                    reduced_content.append(task_instruction)
                    continue_flag = True
                continue
            else:
                # print("here2")
                # instruction should contain english words only : "\u2328\ufe0f Press: $shift$ + A \u2328\ufe0f Type: UG$space$8"
                item.instruction = re.sub(r"\\u[0-9A-Fa-f]{4}", "", item.instruction)
                reduced_content.append(item)
        except:
            raise ValueError(f"Unknown item:{item} in episode:{episode_id}, step:{step_num}")
    return reduced_content


def build_actions(episode_id, step_num, action, img_size, trace=None):
    actionlist = []
    action_type = get_action_type(action)
    # print(action_type)

    def process_caps_lock(string):
        positions = []
        start = 0
        while True:
            pos = string.find("caps_lock", start)
            if pos == -1:  # No more occurrences
                break
            positions.append(pos)
            start = pos + len("caps_lock")
        content = string[:positions[0]]
        for index, pos in enumerate(positions):
            if index % 2 == 0 and index + 1 < len(positions):
                content += string[pos+len("caps_lock"):positions[index + 1]].upper()
            elif index % 2 == 1 and index + 1 < len(positions):
                content += string[pos+len("caps_lock"):positions[index + 1]]
            elif index % 2 == 0 and index + 1 == len(positions):
                content += string[pos+len("caps_lock"):].upper()
            elif index % 2 == 1 and index + 1 == len(positions):
                content += string[pos+len("caps_lock"):]
        return content


    if action_type != "click" and action_type != "doubleClick" and action_type != "rightClick":
        print(action_type)
    if action_type == "click" or action_type == "doubleClick" or action_type == "rightClick" or action_type == "middleClick":
        try:
            coordinates = action.split("(")[1].split(")")[0]
            # print(coordinates)
            x, y = map(float, coordinates.split(","))
            # print(img_size)
            x = max(0, min(x, img_size[0]))
            y = max(0, min(y, img_size[1]))
            # print(x, y)
            actionlist = [
                PyAutoGUIAction(
                    action_type=action_type,
                    target=None,
                    args={
                        "x": x / img_size[0],
                        "y": y / img_size[1],
                    },
                )
            ]
            # print(actionlist)
        except:
            # print(f"shit:{action}")
            return None
            # raise ValueError(f"Unknown click action: {action}")
    elif action_type == "tripleClick":
        coordinates = action.split("(")[1].split(")")[0]
        x, y = map(float, coordinates.split(","))
        x = max(0, min(x, img_size[0]))
        y = max(0, min(y, img_size[1]))
        actionlist = [
            ComputerAction(
                action_type=action_type,
                args={
                    "x": x / img_size[0],
                    "y": y / img_size[1],
                },
            )
        ]
        # raise ValueError(f"computer action in episode:{episode_id}, step:{step_num}")
    elif action_type == "write":
        whole = action.split("Type: ")[-1]
        stack = []
        current = ""
        contents = []
        actions = []

        for char in whole:
            if char == "$":
                if stack and stack[-1] == "$":
                    # Pop the $ and save the key
                    stack.pop()
                    if current:
                        contents.append(["keys", current])
                        current = ""
                else:
                    # If we have accumulated any text before this $, save it
                    if current:
                        if "caps_lock" in current:
                            current = process_caps_lock(current)
                        contents.append(["text", current])
                        current = ""
                    stack.append("$")
            else:
                current += char

        # Handle any remaining content
        if current:
            if "caps_lock" in current: 
                current = process_caps_lock(current)
            contents.append(["text", current])
        


        for content in contents:
            if content[0] == "keys":
                actions.append(
                    PyAutoGUIAction(
                        action_type="press",
                        target=None,
                        args={"keys": [content[1]]},
                    )
                )
            else:
                actions.append(
                    PyAutoGUIAction(
                        action_type="write",
                        target=None,
                        args={"message": content[1]},
                    )
                )
        actionlist = actions


    elif action_type == "press":
        # example: "\u2328\ufe0f Press: $shift$ + N"
        # extract keys from the string after "Press: "
        # split by " + " or "$"
        action=action.replace('\n', '')
        try:
            keys = [action.split("Press: ")[1]]
        except:
            raise ValueError(f"Unknown press action: {action} in episode:{episode_id}, step:{step_num}")
        if "+" in action:
            keys = action.split("Press: ")[1].split(" + ")
        else:
            print("?????")

        
        print(keys)
        for index in range(len(keys)):
            if keys[index].startswith("$") and keys[index].endswith("$") and keys[index].count("$") == 2:
                keys[index] = keys[index].replace("$", "")
                print(keys)
            if "$" in keys[index]:
                new_keys = keys[index]
                new_keys = new_keys.split("$")
                keys[index] = new_keys[0]
                for new_index, new_key in enumerate(new_keys[1:]):
                    if new_key == "":
                        continue
                    keys.insert(index + 1 + new_index, new_key)
            # if "cmd" in keys[index]:
            #     keys[index] = "ctrl"
        for index, key in enumerate(keys):
            if key == "":
                keys.pop(index)
            
        print(keys)

        def extract_content(keys):
            # print(f"extract_content:{keys}")
            contents = []
            content=''
            for index_key, key in enumerate(keys):
                # print(key)
                # process caps lock
                if "caps_lock" in key:
                    key = process_caps_lock(key)
                if "backspace" == key:
                    if content != '':   
                        content = content[:-1]
                    elif content == '':
                        contents.append(["keys", "backspace"])
                elif "space" == key:
                    content += " "
                elif "enter" == key:
                    if content != '':
                        contents.append(["text", content])
                        content = ''
                    contents.append(["keys", "enter"])
                elif "tab" == key:
                    if content != '':
                        contents.append(["text", content])
                        content = ''
                    contents.append(["keys", "tab"])
                
                else:
                    for index_char, char in enumerate(key):
                        # check if the key is a letter using ascii value
                        if ord(char) >= 65 and ord(char) <= 90 or ord(char) >= 97 and ord(char) <= 122:
                            # print("letter")
                            if index_key == 0 and index_char == 0:
                                # check if the first letter is uppercase
                                if char.isupper():
                                    content = char
                                else:
                                    content = char.upper()
                            elif index_key != 0:
                                content += char
                            else:
                                content += char
                        elif char.isdigit():
                            # print("digit")
                            content += char
                        # add "\"
                        elif char in ["-", "_", "+", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "|", ":", '"', "'", "<", ">", "?", "/", "=", "–", ".", ",", ";", "'", "`", ".", "=", "\\", "）", "（", "！", " ", "，", "、", "。", "โ", "ต", "；", "§"]:
                        # elif char in SPECIAL_SYMBOLS:
                            # print("symbol")
                            content += char
                        else:
                            raise ValueError(f"Unknown key({char}) in keys:{keys} in episode:{episode_id}, step:{step_num}")
            if content != '':
                contents.append(["text", content])
            return contents

        # if keys is shift + a series of letter or '-','=', '`' or digit, turn into write uppercase
        try:
            assert len(keys) > 0
        except:
            raise ValueError(f"Unknown keys:{keys} in action:{action} in episode:{episode_id}, step:{step_num}")
        

        if keys[0] == "shift" and "cmd" not in keys and "ctrl" not in keys:
            if len(keys) == 1:
                actionlist = [
                    PyAutoGUIAction(
                        action_type="hotkey",
                        target=None,
                        args={"keys": ["shift"]},
                    )
                ]
            
            elif len(keys) == 2 and keys[1] == "enter":
                actionlist = [
                    PyAutoGUIAction(
                        action_type="hotkey",
                        target=None,
                        args={"keys": ["shift", "enter"]},
                    )
                ]
            
            elif keys[1][0].isalpha() or keys[1][0].isdigit() or keys[1][0] in ["_", "=", "+", "~", "!", "@", "#", "$", "%", "^", "&", "*", "(", ")", "[", "]", "{", "}", "|", ":", "-", '"', "'", "<", ">", "?", "/", ".", "\\", "）", "（", "！", "，", "、", "。", "โ", "ต", "；", "§"]:
            # elif keys[1][0] in SPECIAL_SYMBOLS:
                contents = extract_content(keys[1:])
                print(contents)
                if contents == []:
                    if len(actionlist) == 0:
                        actionlist = [
                            PyAutoGUIAction(
                                action_type="hotkey",
                                target=None,
                                args={"keys": ["shift"]},
                            )
                        ]
                else:
                    for content in contents:
                        if content[0] == "text":
                            if len(actionlist) ==0:
                                actionlist = [
                                    PyAutoGUIAction(
                                        action_type="write",
                                        target=None,
                                        args={"message": content[1]},
                                    )
                                ]
                            else:
                                actionlist.append(
                                    PyAutoGUIAction(
                                        action_type="write",
                                        target=None,
                                        args={"message": content[1]},
                                    )
                                )
                        elif content[0] == "keys":
                            if len(actionlist) ==0:
                                actionlist = [
                                    PyAutoGUIAction(
                                        action_type="press",
                                        target=None,
                                        args={"keys": [content[1]]},
                                    )
                                ]
                            else:
                                actionlist.append(
                                    PyAutoGUIAction(
                                        action_type="press",
                                        target=None,
                                        args={"keys": [content[1]]},
                                    )
                                )
                
                # last_action = None
                # first_action = None
                # if keys[-1] == "enter":
                #     keys = keys[:-1]
                #     last_action = PyAutoGUIAction(
                #         action_type="press",
                #         target=None,
                #         args={"keys": ["enter"]},
                #     )
                # if keys[1] == "enter":
                #     keys = keys[1:]
                #     first_action = PyAutoGUIAction(
                #         action_type="press",
                #         target=None,
                #         args={"keys": ["enter"]},
                #     )
                # actionlist = [
                #     PyAutoGUIAction(
                #         action_type="write",
                #         target=None,
                #         args={"content": extract_content(keys[1:])},
                #     )
                # ]
                # print(extract_content(keys[1:]))
                # if last_action:
                #     actionlist = [last_action] + actionlist
                # if first_action:
                #     actionlist = actionlist + [first_action]
        
        
        else:
            actionlist = [
                PyAutoGUIAction(
                    action_type="hotkey",
                    target=None,
                    args={
                        "keys": keys,
                    },
                )
            ]
            # print(actionlist)

    elif action_type == "dragTo":
        # example: "Drag from (1025.5234375, 362.77734375) to (1711.9765625, 362.77734375)"
        [from_coor, target_coor] = action.split("Drag from ")[1].split(" to ")
        from_x, from_y = map(float, from_coor.split("(")[1].split(")")[0].split(","))
        to_x, to_y = map(float, target_coor.split("(")[1].split(")")[0].split(","))
        actionlist = [
            PyAutoGUIAction(
                action_type="moveTo",
                target=None,
                args={
                    "x": max(0, min(from_x / img_size[0], 1)),
                    "y": max(0, min(from_y / img_size[1], 1)),
                },
            ),
            PyAutoGUIAction(
                action_type="dragTo",
                target=None,
                args={
                    "x": max(0, min(to_x / img_size[0], 1)),
                    "y": max(0, min(to_y / img_size[1], 1)),
                    "button": "left",
                },
            ),
        ]
    elif action_type == "scroll":
        actionlist = []
        # print(trace)
        if trace is not None:
            x, y = max(0, min(trace[0]["x"], img_size[0])), max(0, min(trace[0]["y"], img_size[1]))
            dx, dy = 0, 0
            for action in trace:
                dx += action["dx"]
                dy += action["dy"]
            actionlist.append(
                PyAutoGUIAction(
                    action_type="moveTo",
                    target=None,
                    args={
                        "x": max(0, min(x / img_size[0], 1)),
                        "y": max(0, min(y / img_size[1], 1)),
                    },
                )
            )
            if dx != 0:
                actionlist.append(
                    PyAutoGUIAction(
                        action_type="hscroll",
                        target=None,
                        args={
                            "clicks": dx,
                        },
                    )
                )
            if dy != 0:
                actionlist.append(
                    PyAutoGUIAction(
                        action_type="scroll",
                        target=None,
                        args={
                            "clicks": dy,
                        },
                    )
                )

        else:
            scroll_directions = parse_scroll_to_cardinal(action)
            assert not (scroll_directions["up"] == 0 and scroll_directions["down"] == 0 and scroll_directions["left"] == 0 and scroll_directions["right"] == 0), f"Unknown scroll action: {action} at step{step_num} in episode({episode_id})"
            if scroll_directions["up"] != 0 or scroll_directions["down"] != 0 \
                and abs((scroll_directions["up"] - scroll_directions["down"])) >= 1:
                actionlist.append(
                    PyAutoGUIAction(
                        action_type="scroll",
                        target=None,
                        args={"clicks": scroll_directions["up"] - scroll_directions["down"]},
                    )
                )
            if scroll_directions["left"] != 0 or scroll_directions["right"] != 0 \
                and abs((scroll_directions["right"] - scroll_directions["left"])) >= 2:
                actionlist.append(
                    PyAutoGUIAction(
                        action_type="hscroll",
                        target=None,
                        args={"clicks": scroll_directions["right"] - scroll_directions["left"]},
                    )
                )
    
    elif action_type == "terminate":
        actionlist = [
            ComputerAction(
                action_type="terminate",
                target=None,
                args={
                    "status": "success",
                },
            )
        ]

    else:
        raise ValueError(f"Unknown action: {action}")
    
    try:
        # print(actionlist)
        reduced_actionlist = reduce_actions(actionlist)
        # print(reduced_actionlist)
    except:
        # print(actionlist)
        raise ValueError(f"Unknown actionlist: {action}")

    return reduced_actionlist


def convert_examples(sample_raw):
    trajs = []
    # print("haah")

    skip_episode = [
        # "20241015154629_sebmendoza021@gmail.com_00062806-6dbf-4a4c-9237-2cd3331cc2c0",
        # "20241009100302_keyingl638@gmail.com_0359205f-4c82-426c-8d22-bcf84f3970ce",
        # "20241011191713_zhangqh23@mails.tsinghua.edu.cn_2fc6c64b-02ca-4c1b-a155-8b8cfd027591",
        "20241016155654_prolific_test_3805_72eb88c4-ff2b-45a5-a819-e0fef670779b",
        "20240924144854_tianbaoxiexxx@gmail.com_0b23f6ee-5cc2-4f65-950f-a5a16115b8fc",
        "20241012203145_prolific_test_626_2c27d6bd-dbe9-4919-9393-aab6b826ecb9",
        ###hahahaa
        "20240929152414_martinshin95@gmail.com_4255b742-5eff-45f0-808b-820a219e23ca",
        "20241116123450_samsiyahsamok@gmail.com_ae9478d3-e16f-404d-91db-adc8dee1adae",
        "20241022021432_prolific_test_8771_9b2a7f02-b0f4-4922-a9d8-c89d3adb0cfb"
    ]

    for item in sample_raw:
        episode_id = item["episode_id"]
        # print(episode_id)
        if episode_id in skip_episode:
            continue
        task_instruction = item["task_name"]
        try:
            step_num = len(item["events"])
        except:
            print(episode_id)
            continue
        img_size = [item["metadata"]["screen_width"], item["metadata"]["screen_height"]]
        item["events"] = preprocess_events(item["events"])
        print(step_num)
        content = None
        for i in range(step_num):
            # print(i)
            if i == 0:
                try:
                    content = [
                        ImageObservation(content=item["events"][i]["frame"], filename=f"{episode_id}_{i}.png", source="os")
                    ]
                except:
                    raise ValueError(f"Error in episode {episode_id}, step {i}")
                content.append(TextObservation(content=task_instruction, source="user"))
            else:
                try:
                    content.append(
                        ImageObservation(content=item["events"][i]["frame"], filename=f"{episode_id}_{i}.png", source="os")
                    )
                except Exception as e:
                    continue
                    raise ValueError(f"Error in episode {episode_id}, step {i}: {e}")
            instruction = item["events"][i]["action"]
            rawaction = item["events"][i]["description"]
            if "trace" in item["events"][i]:
                trace = item["events"][i]["trace"]
            else:
                trace = None
            if build_actions(episode_id, i, rawaction, img_size, trace) != None:
                content.append(GUIAction(instruction=instruction, guiactions=build_actions(episode_id, i, rawaction, img_size, trace)))
            else:
                # print(f"Unknown action: {rawaction} in episode {episode_id}, step {i}")
                raise ValueError(f"Unknown action: {rawaction} in episode {episode_id}, step {i}")
        
        # add a terminate action

        if content is not None:
            content = [TextObservation(content=system_instruction, source="system")] + content
            # print(content)
            reduced_content = reduce_content(episode_id, i, content)
            trajs.append(
                Trajectory(
                    task_id="agentnet",
                    type="end2end",
                    example_id=str(episode_id),
                    content=reduced_content,
                )
            )
        else:
            print(f"Empty content in episode {episode_id}")
            return []
        # break

    return trajs
