# CoTGenerator: Reflective Chain-of-Thought Synthesis for Computer Use Tasks

This repository implements a comprehensive framework for adding reflective long chain-of-thought (CoT) reasoning to computer use task demonstrations. Our approach extends beyond simple action annotation by incorporating error identification, reflection, and corrective reasoning to improve the quality of training data for computer use agents.

## Overview

Incorrect or redundant annotations in human demonstrations are not all bad, as long as we can identify and use them to teach the identification and correction of model errors. Our CoT synthesis framework consists of three main components:

- **🔍 Reflector**: Identifies errors and generates reflection reasoning for each step
- **🧠 Generator**: Produces structured CoT with comprehensive agent reasoning
- **📊 Summarizer**: Evaluates trajectories and refines task descriptions

The framework processes computer use demonstrations to generate rich, meaningful CoTs that significantly improve model reasoning and planning capabilities.

## Architecture

Our pipeline includes three key components:

### 1. Reflector
- Inspects each step for correctness and redundancy
- Compares screenshots before and after actions
- Examines alignment between actions, code, and screenshots
- Generates detailed explanations for errors and state transitions

### 2. Generator
- Conditions on full agent context (reflections, history, goals, screenshots, code)
- Generates structured CoT with observation, thought, and action components
- Incorporates visual cues (red markers and zoomed patches) for coordinate grounding
- Handles both mouse and keyboard actions with specialized prompts

### 3. Summarizer
- Refines vague user goals into precise task objectives
- Scores trajectories for alignment, efficiency, and difficulty
- Provides comprehensive evaluation metrics

## Installation

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate cotgenerator
```

Set up your API key:
```bash
export API_KEY="your-api-key-here"
```

## Input Structure

The system expects input in JSONL format where each line represents a task demonstration:

```json
{
  "task_id": "unique_task_identifier",
  "instruction": "Original task description",
  "traj": [
    {
      "image": "screenshot_filename.png",
      "value": {
        "code": "pyautogui.click(x=0.5, y=0.3)"
      }
    }
  ]
}
```

### Input Fields:
- `task_id`: Unique identifier for the task
- `instruction`: Natural language description of the task goal
- `traj`: List of trajectory steps, each containing:
  - `image`: Screenshot filename showing the state before the action
  - `value.code`: PyAutoGUI code representing the action to be performed

## Usage

### Basic Usage

Generate CoT annotations for your demonstrations:

```bash
python gen_cot.py \
  --traj_path ./gen_cot_example/raw_example.jsonl \
  --image_folder ./gen_cot_example/images \
  --output_dir ./gen_cot_example/output/tasks \
  --model claude-3-7-sonnet-20250219
```

### Advanced Options

```bash
python gen_cot.py \
  --traj_path ./data/trajectories.jsonl \
  --image_folder ./data/images \
  --output_dir ./output/enhanced_tasks \
  --model claude-3-7-sonnet-20250219 \
  --num_threads 4 \
  --max_num 100 \
  --need_double_check \
  --with_prior_judge \
  --no_auto_merge
```

### Parameters:
- `--traj_path`: Path to input JSONL file with task demonstrations
- `--image_folder`: Directory containing screenshot images
- `--output_dir`: Output directory for generated CoT data
- `--model`: LLM model for CoT generation (default: claude-3-7-sonnet-20250219)
- `--num_threads`: Number of parallel processing threads
- `--max_num`: Maximum number of tasks to process
- `--need_double_check`: Enable double-checking of generated content
- `--with_prior_judge`: Use prior judgment results (for Ubuntu dataset)
- `--no_auto_merge`: Disable automatic merging to JSONL after processing

### Visualization

View and analyze the generated results:

```bash
cd gen_cot_example/visualization
python app.py \
  --data_path ../output/tasks \
  --image_folder ../images \
  --port 5000
```

Then open http://localhost:5000 in your browser to explore the enhanced demonstrations.

## Output Structure

The system generates enriched demonstrations with detailed CoT reasoning:

```json
{
  "task_id": "unique_task_identifier",
  "instruction": "Original task description",
  "traj": [
    {
      "index": 0,
      "image": "screenshot_filename.png",
      "value": {
        "code": "pyautogui.click(x=0.5, y=0.3)",
        "observation": "Detailed description of current screen state...",
        "thought": "Agent's reasoning about the situation and next steps...",
        "action": "Clear description of the intended action...",
        "last_step_correct": true,
        "last_step_redundant": false,
        "reflection": "Analysis of the action's effects..."
      }
    }
  ],
  "task_completed": true,
  "alignment_score": 8,
  "efficiency_score": 7,
  "task_difficulty": 6,
  "reason": "Task completed successfully with minor inefficiencies...",
  "actual_task": "Refined task description based on actual actions...",
  "natural_language_task": "Human-friendly version of the task..."
}
```

### Output Fields:

#### Step-level Fields:
- `observation`: Detailed description of the current computer state and relevant elements
- `thought`: Agent's comprehensive reasoning including:
  - State change analysis from previous actions
  - Progress assessment toward task completion
  - Strategic planning for next steps  
  - Justification for the chosen action
- `action`: Clear, actionable instruction aligned with the code
- `last_step_correct`: Boolean indicating if the previous step was executed correctly
- `last_step_redundant`: Boolean indicating if the previous step was unnecessary
- `reflection`: Analysis of the action's effects on the computer state

#### Trajectory-level Fields:
- `task_completed`: Boolean indicating successful task completion
- `alignment_score`: Integer (0-10) measuring how well actions align with the original goal
- `efficiency_score`: Integer (0-10) measuring trajectory efficiency (fewer redundant steps = higher score)
- `task_difficulty`: Integer (0-10) representing inherent task complexity
- `reason`: Textual summary explaining the evaluation reasoning
- `actual_task`: Refined task description based on actual demonstrated actions
- `natural_language_task`: Human-friendly version of the task description

## Key Features

### 🎯 Error Identification & Correction
The reflector component identifies incorrect or redundant actions and provides detailed explanations, helping models learn to recognize and correct errors.

### 🖼️ Visual Grounding
Incorporates visual cues including red markers on action coordinates and zoomed image patches to improve spatial reasoning.

### 🔄 Comprehensive Reflection
Each step includes detailed reflection on state changes, progress assessment, and strategic planning.

### 📈 Multi-dimensional Evaluation
Provides alignment, efficiency, and difficulty scores for comprehensive trajectory assessment.

### 🚀 Scalable Processing
Supports multi-threaded processing for large-scale dataset enhancement.

## File Structure

```
CoTGenerator/
├── gen_cot.py              # Main CoT generation script
├── merge_json.py           # Utility for merging results to JSONL
├── utils.py                # Helper functions for image processing and LLM calls
├── module/
│   ├── generator.py        # CoT generation prompts and parsing
│   ├── evaluator.py        # Trajectory evaluation components
│   ├── reflector.py        # Step-level reflection and error detection
│   └── reflector_with_prior_judge.py  # Alternative reflector with prior judgments
├── gen_cot_example/
│   ├── raw_example.jsonl   # Example input data
│   ├── images/             # Screenshot images
│   ├── output/tasks/       # Generated CoT data
│   └── visualization/      # Web-based result viewer
└── environment.yml         # Conda environment specification
```


## License

This project is licensed under the MIT License - see the LICENSE file for details.