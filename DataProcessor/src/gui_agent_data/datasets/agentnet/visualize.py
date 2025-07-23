import json
import os
# import random

from flask import Flask, render_template_string, request

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Agent Trajectory Viewer</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f0f0f0;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .navigation {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 4px;
        }
        .nav-info {
            font-size: 14px;
            color: #666;
        }
        .nav-buttons {
            display: flex;
            gap: 10px;
        }
        button {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            background-color: #007bff;
            color: white;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        button:hover:not(:disabled) {
            background-color: #0056b3;
        }
        .step {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 4px;
            overflow: hidden;
        }
        .step-header {
            padding: 10px;
            background: #f8f9fa;
            border-bottom: 1px solid #ddd;
            font-weight: bold;
            color: #333;
        }
        .step-content {
            padding: 15px;
        }
        .observation {
            background: #fff;
            padding: 10px;
        }
        .text-observation {
            white-space: pre-wrap;
            font-family: monospace;
            background: #f8f9fa;
            padding: 10px;
            border-radius: 4px;
        }
        .image-container {
            position: relative;
            display: inline-block;
        }
        .action-marker {
            position: absolute;
            pointer-events: none;
        }
        .click-marker {
            width: 10px;  /* Smaller size for a dot */
            height: 10px;
            background: red;
            border-radius: 50%;  /* Makes it circular */
            transform: translate(-50%, -50%);
        }

        .click-marker::before,
        .click-marker::after {
            content: '';
            position: absolute;
            background: red;
        }
        .click-marker::before {
            width: 30px;  /* 增加长度 */
            height: 4px;  /* 增加粗细 */
            top: 13px;
            left: 0;
        }
        .click-marker::after {
            width: 4px;   /* 增加粗细 */
            height: 30px; /* 增加长度 */
            left: 13px;
            top: 0;
        }
        .click-marker::before,
        .click-marker::after {
            display: none;  /* Removes the + shape */
        }
        .click-circle {
            position: absolute;
            width: 40px;
            height: 40px;
            border: 3px solid red;
            border-radius: 50%;
            transform: translate(-50%, -50%);
            pointer-events: none;
        }
        .input-marker {
            background: rgba(0, 128, 255, 0.8);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            transform: translate(-50%, -100%);
            margin-top: -10px;
            max-width: 200px;
            word-wrap: break-word;
        }
        .hotkey-marker {
            background: rgba(255, 165, 0, 0.8);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            transform: translate(-50%, -100%);
            margin-top: -10px;
        }
        .drag-marker {
            width: 10px;
            height: 10px;
            background: orange;
            border-radius: 50%;
            transform: translate(-50%, -50%);
        }
        .drag-line {
            position: absolute;
            height: 2px;
            background: orange;
            transform-origin: left center;
            pointer-events: none;
        }
        .coordinates {
            background: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            transform: translate(-50%, 20px);
        }
        .observation-container {
            position: relative;
            display: inline-block;
        }
        .action-tooltip {
            position: absolute;
            background: rgba(0, 0, 0, 0.8);
            color: white;
            padding: 8px;
            border-radius: 4px;
            font-size: 12px;
            max-width: 300px;
            z-index: 100;
            display: none;
        }
        .action {
            background: #e8f4ff;
            padding: 10px;
            margin-top: 10px;
            border-radius: 4px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        .action-marker:hover + .action-tooltip {
            display: block;
        }
        .nav-controls {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .pagination {
            display: flex;
            gap: 10px;
            justify-content: flex-end;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="navigation">
            <div class="nav-info">
                <strong>Task ID:</strong> {{ trajectories[current_index].task_id }}<br>
                <strong>Example ID:</strong> {{ trajectories[current_index].example_id }}<br>
                <strong>Trajectory:</strong> {{ current_index + 1 }} / {{ trajectories|length }}<br>
                <strong>Page:</strong> {{ current_page }} / {{ total_pages }} (Total files: {{ total_files }})
            </div>
            <div class="nav-controls">
                <div class="nav-buttons">
                    <button onclick="changePage(-1)" {% if current_index == 0 %}disabled{% endif %}>Previous</button>
                    <button onclick="changePage(1)" {% if current_index == trajectories|length - 1 %}disabled{% endif %}>Next</button>
                </div>
                <div class="pagination">
                    <button onclick="changeBatch(-1)" {% if current_page == 1 %}disabled{% endif %}>Previous Batch</button>
                    <button onclick="changeBatch(1)" {% if current_page == total_pages %}disabled{% endif %}>Next Batch</button>
                </div>
            </div>
        </div>

        {% for step in trajectories[current_index].content %}
            <div class="step">
                <div class="step-header">
                    {% if loop.index == 1 %}
                        System Prompt
                    {% elif loop.index == 2 %}
                        Obs 0
                    {% elif loop.index == 3 %}
                        Task Instruction
                    {% else %}
                        {% if loop.index % 2 == 0 %}
                            Step {{ ((loop.index - 4) / 2) | int }}
                        {% else %}
                            Obs {{ ((loop.index - 3) / 2) | int }}
                        {% endif %}
                    {% endif %}
                </div>
                <div class="step-content">
                    {% if step.class_ == 'text_observation' %}
                        <div class="observation">
                            <div class="text-observation">{{ step.content }}</div>
                            <div class="meta">Source: {{ step.source }}</div>
                        </div>
                    {% elif step.class_ == 'image_observation' %}
                        <div class="observation">
                            <div class="observation-container">
                                <img src="data:image/png;base64,{{ step.content.split(',')[1] if ',' in step.content else step.content }}">

                                {# 查找下一个action并标记在当前图像上 #}
                                {% set found_action = namespace(value=false) %}
                                {% for next_step in trajectories[current_index].content[loop.index0 + 1:] %}
                                    {% if not found_action.value and next_step.guiactions is defined and next_step.guiactions %}
                                        {% set found_action.value = true %}
                                        {% for action in next_step.guiactions %}
                                            {% if action.args %}
                                                {% if (action.action_type in ['click', 'doubleClick', 'rightClick', 'moveTo', 'dragTo']) and 'x' in action.args and 'y' in action.args %}
                                                    <div class="action-marker click-marker"
                                                         style="left: {{ action.args.x * 100 }}%; top: {{ action.args.y * 98 }}%">
                                                    </div>
                                                    <div class="coordinates"
                                                         style="left: {{ action.args.x * 100 }}%; top: {{ action.args.y * 98 }}%">
                                                        ({{ "%.3f"|format(action.args.x) }}, {{ "%.3f"|format(action.args.y) }})
                                                    </div>
                                                {% elif action.action_type == 'drag' and 'start_x' in action.args and 'start_y' in action.args and 'end_x' in action.args and 'end_y' in action.args %}
                                                    {# Start point marker #}
                                                    <div class="action-marker drag-marker"
                                                         style="left: {{ action.args.start_x * 100 }}%; top: {{ action.args.start_y * 100 }}%">
                                                    </div>
                                                    <div class="coordinates"
                                                         style="left: {{ action.args.start_x * 100 }}%; top: {{ action.args.start_y * 100 }}%">
                                                        ({{ "%.3f"|format(action.args.start_x) }}, {{ "%.3f"|format(action.args.start_y) }})
                                                    </div>

                                                    {# End point marker #}
                                                    <div class="action-marker drag-marker"
                                                         style="left: {{ action.args.end_x * 100 }}%; top: {{ action.args.end_y * 100 }}%">
                                                    </div>
                                                    <div class="coordinates"
                                                         style="left: {{ action.args.end_x * 100 }}%; top: {{ action.args.end_y * 100 }}%">
                                                        ({{ "%.3f"|format(action.args.end_x) }}, {{ "%.3f"|format(action.args.end_y) }})
                                                    </div>

                                                    {# Line connecting the points #}
                                                    {% set dx = action.args.end_x - action.args.start_x %}
                                                    {% set dy = action.args.end_y - action.args.start_y %}
                                                    {% set length = (dx * dx + dy * dy) ** 0.5 %}
                                                    {% set angle = (dy/dx)|atan2 * 180 / 3.14159 if dx != 0 else 90 %}
                                                    <div class="drag-line"
                                                         style="left: {{ action.args.start_x * 100 }}%;
                                                                top: {{ action.args.start_y * 100 }}%;
                                                                width: {{ length * 100 }}%;
                                                                transform: translate(-0%, -50%) rotate({{ angle }}deg);">
                                                    </div>
                                                {% elif action.action_type == 'type' and 'x' in action.args and 'y' in action.args %}
                                                    <div class="action-marker input-marker"
                                                         style="left: {{ action.args.x * 100 }}%; top: {{ action.args.y * 100 }}%">
                                                        ⌨️ "{{ action.args.text }}"
                                                    </div>
                                                {% elif action.action_type == 'hotkey' and 'x' in action.args and 'y' in action.args %}
                                                    <div class="action-marker hotkey-marker"
                                                         style="left: {{ action.args.x * 100 }}%; top: {{ action.args.y * 100 }}%">
                                                        ⌨️ {{ action.args.keys }}
                                                    </div>
                                                {% endif %}
                                                <div class="action-tooltip">
                                                    {% if next_step.observation %}
                                                        <div><strong>Observation:</strong> {{ next_step.observation }}</div>
                                                    {% endif %}
                                                    {% if next_step.thought %}
                                                        <div><strong>Thought:</strong> {{ next_step.thought }}</div>
                                                    {% endif %}
                                                    {% if next_step.instruction %}
                                                        <div><strong>Instruction:</strong> {{ next_step.instruction }}</div>
                                                    {% endif %}
                                                </div>
                                            {% endif %}
                                        {% endfor %}
                                    {% endif %}
                                {% endfor %}
                            </div>
                        </div>
                    {% endif %}

                    {% if step.guiactions is defined and step.guiactions %}
                        <div class="action">
                            {% if step.observation %}
                                <div><strong>Observation:</strong> {{ step.observation }}</div>
                            {% endif %}
                            {% if step.thought %}
                                <div><strong>Thought:</strong> {{ step.thought }}</div>
                            {% endif %}
                            {% if step.instruction %}
                                <div><strong>Instruction:</strong> {{ step.instruction }}</div>
                            {% endif %}
                            <div><strong>Actions:</strong></div>
                            {% for action in step.guiactions %}
                                <div class="action-detail">
                                    Type: {{ action.action_type }}
                                    {% if action.args %}
                                        <br>Arguments: {{ action.args | tojson }}
                                    {% endif %}
                                </div>
                            {% endfor %}
                        </div>
                    {% endif %}
                </div>
            </div>
        {% endfor %}
    </div>

    <script>
        function changePage(delta) {
            const currentIndex = {{ current_index }};
            const newIndex = currentIndex + delta;
            const currentPage = {{ current_page }};
            window.location.href = `/?index=${newIndex}&page=${currentPage}`;
        }

        function changeBatch(delta) {
            const currentPage = {{ current_page }};
            const newPage = currentPage + delta;
            window.location.href = `/?page=${newPage}&index=0`;
        }
    </script>
</body>
</html>
"""


def process_data(json_path):
    with open(json_path) as f:
        data = json.load(f)
        return data[:10]


@app.route("/")
def index():
    # dir_path = "standardized_data_dir or inner_monologue_data_dir"
    dir_path = "/cpfs03/data/shared/Group-m6/ludunjie.ldj/OpenCUA/gui-agent-data/datasets/agentnet/standardized"
    files = sorted(os.listdir(dir_path))  # Sort files for consistent ordering
    
    # Get pagination parameters
    page = int(request.args.get("page", 1))
    per_page = 10
    total_files = len(files)
    total_pages = (total_files + per_page - 1) // per_page
    
    # Ensure page is within valid range
    page = max(1, min(page, total_pages))
    
    # Get files for current page
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, total_files)
    current_files = files[start_idx:end_idx]
    
    # Load trajectories for current page
    trajectories = [json.load(open(os.path.join(dir_path, f))) for f in current_files]

    # Get current trajectory index
    current_index = int(request.args.get("index", 0))
    current_index = max(0, min(current_index, len(trajectories) - 1))

    return render_template_string(
        HTML_TEMPLATE, 
        trajectories=trajectories, 
        current_index=current_index,
        current_page=page,
        total_pages=total_pages,
        total_files=total_files
    )


if __name__ == "__main__":
    app.run(debug=True, port=5678)
