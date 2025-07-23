import math

from PIL import Image, ImageDraw, ImageFont


def parse_box(box_str, keep_float=False, split_token=","):
    """Parse the box string into x1, y1, x2, y2 coordinates.

    input: <box>x1, y1, x2, y2</box>
    output: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box_str.split("<box>")[-1].split("</box>")[0].strip().split(split_token)
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if keep_float:
        return x1, y1, x2, y2
    else:
        return int(x1), int(y1), int(x2), int(y2)


def parse_point(point_str, keep_float=False, split_token=","):
    """Parse the point string into x, y coordinates.

    input: <point>x, y</point>
    output: (x, y)
    """
    x, y = point_str.split("<point>")[-1].split("</point>")[0].strip().split(split_token)
    x, y = float(x), float(y)
    if keep_float:
        return x, y
    else:
        return int(x), int(y)


def draw_rectangle(
    draw, box_coords, width=2, outline_color=(0, 255, 0), is_fill=False, bg_color=(0, 255, 0), transparency=50
):
    if is_fill:
        # Calculate the alpha value based on the transparency percentage
        alpha = int((1 - transparency / 100) * 255)

        # Set the fill color with the specified background color and transparency
        fill_color = tuple(bg_color) + (alpha,)

        draw.rectangle(box_coords, width=width, outline=outline_color, fill=fill_color)
    else:
        draw.rectangle(box_coords, width=width, outline=outline_color)


def draw_circle(
    draw, center, radius=10, width=2, outline_color=(0, 255, 0), is_fill=False, bg_color=(0, 255, 0), transparency=80
):
    # Calculate the bounding box coordinates for the circle
    x1 = center[0] - radius
    y1 = center[1] - radius
    x2 = center[0] + radius
    y2 = center[1] + radius
    bbox = (x1, y1, x2, y2)

    # Draw the circle
    if is_fill:
        # Calculate the alpha value based on the transparency percentage
        alpha = int((1 - transparency / 100) * 255)

        # Set the fill color with the specified background color and transparency
        fill_color = tuple(bg_color) + (alpha,)

        draw.ellipse(bbox, width=width, outline=outline_color, fill=fill_color)
    else:
        draw.ellipse(bbox, width=width, outline=outline_color)


def draw_text_with_bg_box(
    draw,
    text,
    view_port,
    position,
    font_size=24,
    font_color=(0, 0, 0),
    bg_padding=16,
    bg_color=(179, 238, 58),
    max_width=1000,
):

    # Define the font and size for the text
    # if platform.system() == "Windows":
    #     font = ImageFont.truetype("arial.ttf", font_size)
    # else:
    #     font = ImageFont.truetype("Arial.ttf", font_size)
    font = ImageFont.load_default()

    if len(text) > 500:
        text = text[:500] + "..."

    # Split text into multiple lines if it exceeds the max width
    lines = []
    words = text.split()
    current_line = words[0]
    for word in words[1:]:
        # Check the width of the current line with the next word
        text_bbox = draw.textbbox((0, 0), current_line + " " + word, font=font)
        if text_bbox[2] - text_bbox[0] <= max_width:
            current_line += " " + word
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)

    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_height = 0
    text_width = 0
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)
        text_width = max(text_width, text_bbox[2] - text_bbox[0])
        text_height += text_bbox[3] - text_bbox[1] + bg_padding // 2

    # Define the position of the text based on the specified position parameter
    image_width, image_height = view_port
    if position == "top-left":
        text_x = image_width * 0.02
        text_y = image_height * 0.05
    elif position == "bottom-middle":
        text_x = (image_width - text_width) // 2
        text_y = max(0, image_height * 0.95 - text_height)
    elif position == "top-middle":
        text_x = (image_width - text_width) // 2
        text_y = image_height * 0.05
    elif position.startswith("point"):
        text_x, text_y = position.split("-")[1:]
        text_x, text_y = int(text_x), int(text_y)
    else:
        print("unsupported position")

    # Draw the background box
    draw_rectangle(
        draw,
        [(text_x, text_y), (text_x + text_width + bg_padding, text_y + text_height + bg_padding)],
        outline_color=(154, 205, 50),
        is_fill=True,
        bg_color=bg_color,
    )

    # Draw the text on top of the background box
    current_y = text_y + bg_padding // 2
    for line in lines:
        draw.text((text_x + bg_padding // 2, current_y), line, font=font, fill=font_color)
        current_y += (
            draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] + bg_padding // 2
        )


def draw_index_with_bg_box(
    draw, text, position, font_size=18, font_color=(255, 255, 255), bg_padding=20, bg_color=(66, 119, 56)
):
    # Define the font and size for the text
    # if platform.system() == "Windows":
    #     font = ImageFont.truetype("arial.ttf", font_size)
    # else:
    #     font = ImageFont.truetype("Arial.ttf", font_size)
    font = ImageFont.load_default()

    # Calculate the bounding box of the text
    text_bbox = draw.textbbox((0, 0), text, font_size=font_size)

    # Extract the width and height from the bounding box
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Define the position of the text based on the specified position parameter
    text_x, text_y = position[0], max(0, position[1] - font_size // 2 - 5)

    # Draw the background box
    draw_rectangle(
        draw,
        [(text_x, text_y), (text_x + text_width + bg_padding, text_y + text_height + bg_padding + 10)],
        outline_color=bg_color,
        is_fill=True,
        bg_color=bg_color,
    )

    # Draw the text on top of the background box
    draw.text((text_x, text_y), text, font=font, fill=font_color)


def draw_point(draw, center, radius1=3, radius2=12, color=(0, 255, 0), width=2):
    draw_circle(draw, center, radius=radius1, outline_color=color, width=width * 2)
    draw_circle(draw, center, radius=radius2, outline_color=color, width=width)


def draw_line_with_arrow(draw, start_point, end_point, color=(0, 255, 0), width=3, arrow_size=10):
    # Draw the line
    x1, y1 = start_point
    x2, y2 = end_point
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)

    # Compute the angle between the line and the x-axis
    angle = math.atan2(y2 - y1, x2 - x1)

    # Calculate coordinates for arrowhead
    x_arrow = x2 - arrow_size * math.cos(angle + math.pi / 6)
    y_arrow = y2 - arrow_size * math.sin(angle + math.pi / 6)
    x_arrow2 = x2 - arrow_size * math.cos(angle - math.pi / 6)
    y_arrow2 = y2 - arrow_size * math.sin(angle - math.pi / 6)

    # Draw arrowhead
    draw.polygon([(x2, y2), (x_arrow, y_arrow), (x_arrow2, y_arrow2)], fill=color)


def actions_visual(actions: list, pil_img: Image, ins_cmd: str, color=(255, 48, 48), from_eval=False) -> Image:
    draw = ImageDraw.Draw(pil_img)

    image_h = pil_img.height
    image_w = pil_img.width

    # if isinstance(action_group, dict):
    #     action_group = [action_group]

    name_group = ""
    last_action = None
    for i, action in enumerate(actions):
        name_group += "{}. {}".format(i + 1, action["action_type"])

        if action["args"] and "x" in action["args"] and "y" in action["args"]:
            center = action["args"]["x"] * image_w, action["args"]["y"] * image_h
            draw_point(draw, center, color=color)

        if action["action_type"] == "dragTo" and last_action is not None and last_action["action_type"] == "moveTo":
            start_point = last_action["args"]["x"] * image_w, last_action["args"]["y"] * image_h
            end_point = action["args"]["x"] * image_w, action["args"]["y"] * image_h
            draw_line_with_arrow(draw, start_point, end_point, color=color)

        # Draw element with index
        if action["action_type"] in {"click", "rightClick", "doubleClick", "moveTo", "dragTo"} and action["target"]:

            if "absolute_bbox" in action["target"]:
                box_coords = action["target"]["absolute_bbox"]
            else:
                relative_box_coords = action["target"]["bbox"]
                box_coords = [
                    relative_box_coords[0] * image_w,
                    relative_box_coords[1] * image_h,
                    relative_box_coords[2] * image_w,
                    relative_box_coords[3] * image_h,
                ]

            draw_rectangle(draw, box_coords, outline_color=color)

            draw_index_with_bg_box(
                draw, str(i + 1), (box_coords[0], box_coords[1]), font_size=image_h // 60, bg_padding=image_h // 480
            )
            if "text" in action["target"] and action["target"]["text"]:
                name_group += ' "' + action["target"]["text"] + '"'

        # Draw scroll
        if action["action_type"] == "scroll":
            dx, dy, x, y = (
                action["args"]["dx"] * image_w,
                action["args"]["dy"] * image_h,
                action["args"]["x"] * image_w,
                action["args"]["y"] * image_h,
            )
            start_point = x, y
            end_point = x + dx, y + dy

            draw_line_with_arrow(draw, start_point, end_point, color=color)

        if action["action_type"] == "write" or action["action_type"] == "typewrite":
            name_group += ' "' + action["args"]["text"].replace("\n", "\t") + '"'

        if action["action_type"] == "press":
            name_group += ' "' + action["args"]["key"] + '"'

        if action["action_type"] == "hotkey":
            name_group += ' "' + "+".join(action["args"]["keys"]) + '"'

        last_action = action
        name_group += "\n"

    # Draw action_names
    draw_text_with_bg_box(
        draw, text=name_group, view_port=(image_w, image_h), position="top-left", font_size=image_h // 50
    )

    # Draw instruction
    if ins_cmd != "":
        draw_text_with_bg_box(
            draw,
            text=ins_cmd,
            view_port=(image_w, image_h),
            position="bottom-middle",
            font_size=image_h // 50,
            max_width=int(image_w * 0.8),
        )

    return pil_img
