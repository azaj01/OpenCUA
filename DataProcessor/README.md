# gui-agent-data

A Python package for processing and standardizing AgentNet GUI agent trajectory data.

## Overview

This package provides a specialized pipeline for processing AgentNet GUI agent interaction data, converting raw AgentNet trajectory recordings into a standardized format suitable for training and evaluation of GUI automation agents. The package handles video processing, event extraction, and trajectory standardization specifically for the AgentNet dataset format.

## Installation

### Prerequisites
- Python 3.11
- PDM (Python Dependency Manager) or pip

### Setup

1. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

2. **Alternative: Using PDM (recommended for development):**
   ```bash
   pdm install
   ```

## Quick Start

The data processing pipeline consists of two main stages:

### 1. Extract Raw Data
```bash
# Extract all samples from AgentNet dataset
xvfb-run -a ./scripts/extract_raw.sh agentnet -1

# Extract specific number of samples
xvfb-run -a ./scripts/extract_raw.sh agentnet 10
```

### 2. Convert to Standardized Format
```bash
# Convert all raw trajectories to standardized format
xvfb-run -a ./scripts/raw_to_standardized.sh agentnet -1

# Convert specific number of samples
xvfb-run -a ./scripts/raw_to_standardized.sh agentnet 10
```

**Note:** `xvfb-run -a` is used to run GUI-related processing in a virtual display environment, which is especially useful on headless servers.

## Project Structure

```
DataProcessor/
├── src/gui_agent_data/           # Main package
│   ├── datasets/agentnet/       # AgentNet dataset processors
│   │   ├── extract_raw.py       # Raw AgentNet data extraction
│   │   ├── raw_to_standardized.py # AgentNet standardization
│   │   └── visualize.py         # AgentNet visualization
│   ├── schema/                   # Data schemas
│   │   ├── action/              # Action type definitions
│   │   ├── observation/         # Observation type definitions
│   │   └── trajectory.py        # Trajectory schema
│   ├── stage/                   # Processing pipeline stages
│   │   ├── extract_raw.py       # Generic extraction wrapper
│   │   └── raw_to_standardized.py # Generic standardization wrapper
│   └── utils/                   # Utility functions
├── datasets/agentnet/           # AgentNet processed data
│   ├── raw/                     # Raw extracted AgentNet data
│   ├── raw_trajs/              # Raw trajectory files
│   └── standardized/            # Standardized trajectory format
├── scripts/                     # Processing scripts
│   ├── extract_raw.sh          # AgentNet extraction script
│   └── raw_to_standardized.sh  # AgentNet standardization script
└── pyproject.toml              # Package configuration
```

## Data Schema

The package defines standardized schemas for GUI agent data:

### Actions
- **GUIAction**: GUI interactions (clicks, typing, etc.)
- **ApiAction**: API calls
- **CodeAction**: Code execution
- **MessageAction**: Text/chat messages

### Observations
- **ImageObservation**: Screenshots and visual data
- **TextObservation**: Text-based observations
- **MultimodalObservation**: Combined image and text data

### Trajectory
A trajectory contains:
- `task_id`: Unique task identifier
- `example_id`: Optional example identifier
- `type`: Either "grounding" or "end2end"
- `content`: List of actions and observations

## AgentNet Data Processing

This package specifically handles AgentNet dataset processing:

- **Video Processing**: Extracts frames from recorded GUI interaction videos
- **Event Extraction**: Processes interaction events, HTML snapshots, and accessibility trees
- **Metadata Handling**: Extracts task information and session metadata
- **Standardization**: Converts AgentNet format to unified trajectory schema

### AgentNet Data Structure
The raw AgentNet data includes:
- Video recordings (`.mp4`) of GUI interactions
- Event logs (`events.jsonl`, `reduced_events_complete.jsonl`)
- HTML snapshots (`html.jsonl`)
- Accessibility trees (`a11y.jsonl`)
- Element information (`element.jsonl`)
- Task metadata (`task_name.json`, `metadata.json`)

## Development

### Code Quality

The project uses:
- **Ruff**: For linting and code formatting
- **Pydantic**: For data validation and schemas
- **Pre-commit hooks**: For automated code quality checks

Run linting:
```bash
ruff check .
ruff format .
```

## Dependencies

Key dependencies include:
- `pydantic`: Data validation and schemas
- `opencv-python`: Video/image processing
- `pyautogui`: GUI automation utilities
- `tqdm`: Progress bars
- `pandas`: Data manipulation
- `pillow`: Image processing

See `pyproject.toml` for the complete list of dependencies.

## Usage Examples

### Processing AgentNet Data Programmatically
```python
# Extract raw AgentNet examples
from gui_agent_data.datasets.agentnet.extract_raw import get_raw_examples

raw_examples = get_raw_examples(num_samples=5)
for example in raw_examples:
    print(f"Episode ID: {example['episode_id']}")
    print(f"Task: {example.get('task_name', 'N/A')}")
```

### Loading Standardized AgentNet Trajectories
```python
import json
from gui_agent_data.schema.trajectory import Trajectory

# Load a standardized AgentNet trajectory
with open('datasets/agentnet/standardized/example.json', 'r') as f:
    data = json.load(f)
    trajectory = Trajectory(**data)
    
print(f"Task ID: {trajectory.task_id}")
print(f"Trajectory type: {trajectory.type}")
print(f"Number of steps: {len(trajectory.content)}")

# Inspect trajectory content
for i, step in enumerate(trajectory.content):
    print(f"Step {i}: {type(step).__name__}")
```

## Contributing

1. Install development dependencies: `pdm install`
2. Set up pre-commit hooks: `pre-commit install`
3. Make changes following the existing code style
4. Run tests and linting before submitting PRs

## License

MIT License
