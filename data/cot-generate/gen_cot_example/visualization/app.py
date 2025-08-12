
from flask import Flask, render_template, send_from_directory, jsonify, send_file
import json, os, glob, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Thread lock for safe concurrent access to shared data structures
lock = threading.Lock()   

def _scan_subdir(subdir: str):
    """
    Scan a single subdirectory to check if it contains valid task data.
    
    Args:
        subdir (str): Name of the subdirectory to scan
        
    Returns:
        tuple or None: Returns (task_id, task_path) if valid task found, None otherwise
        
    A valid task directory must:
    1. Be an actual directory
    2. Contain a 'meta.json' file
    3. Contain at least one JSON file other than 'meta.json'
    """
    task_path = os.path.join(DATA_PATH, subdir)
    meta_file = os.path.join(task_path, 'meta.json')
    
    # Check if directory exists and contains meta.json
    if not (os.path.isdir(task_path) and os.path.exists(meta_file)):
        return None
        
    # Find step files (JSON files excluding meta.json)
    step_files = [f for f in glob.glob(os.path.join(task_path, '*.json'))
                  if os.path.basename(f) != 'meta.json']
    if not step_files:
        return None
        
    # Read meta.json to get task_id
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tid = str(meta.get('task_id', subdir))
    return tid, task_path


# Initialize Flask application
app = Flask(__name__)

# Command line argument parser setup
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='./OpenCUA/CoTGenerator/gen_cot_example/output/tasks/',
                    help='Path to data directory or JSONL file containing tasks')
parser.add_argument('--image_folder', type=str,
                    default="./OpenCUA/CoTGenerator/gen_cot_example/images",
                    help='Path to folder containing task-related images')
parser.add_argument('--port', type=int, default=5000,
                    help='Port number for the Flask server')
args = parser.parse_args()

# Global configuration variables
DATA_PATH = os.path.abspath(args.data_path)      # 转换为绝对路径
IMAGE_FOLDER = os.path.abspath(args.image_folder) # 转换为绝对路径
print(f"Absolute IMAGE_FOLDER: {IMAGE_FOLDER}")

# Global data structures for task indexing
task_index = {}      # Maps task_id -> {'type': 'jsonl'/'dir', 'ptr': line_number/path}
task_keys = []       # Ordered list of task_ids to maintain sequence
total_tasks = 0      # Total number of tasks found


def init_task_index():
    """
    Initialize the task index by scanning the data source.
    Supports two data formats:
    1. JSONL file: Each line contains a complete task as JSON
    2. Directory structure: Each subdirectory contains meta.json + step files
    
    Uses multithreading for efficient directory scanning.
    Updates global variables: task_index, task_keys, total_tasks
    """
    global task_index, total_tasks

    # Case 1: DATA_PATH is a JSONL file
    if os.path.isfile(DATA_PATH):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                data = json.loads(line)
                tid = str(data.get('task_id', line_no))  # Use line number as fallback ID
                task_index[tid] = {'type': 'jsonl', 'ptr': line_no}
                task_keys.append(tid)
                
    # Case 2: DATA_PATH is a directory containing task subdirectories
    else:
        subdirs = sorted(os.listdir(DATA_PATH))
        
        # Use thread pool for concurrent directory scanning
        with ThreadPoolExecutor(max_workers=128) as pool:
            # Submit scanning tasks for all subdirectories
            futures = [pool.submit(_scan_subdir, sd) for sd in subdirs]
            
            # Process completed tasks with progress bar
            for fut in tqdm(as_completed(futures), total=len(futures)):
                result = fut.result()
                if result:
                    tid, task_path = result
                    # Thread-safe update of shared data structures
                    with lock:
                        task_index[tid] = {'type': 'dir', 'ptr': task_path}
                        task_keys.append(tid)

    total_tasks = len(task_index)
    print(f'Indexing complete: Found {total_tasks} tasks')


def load_task_by_id(identifier: str):
    """
    Load a complete task by its identifier.
    
    Args:
        identifier (str): Either a task_id string or numeric index (e.g., '0', '1')
        
    Returns:
        dict or None: Complete task data if found, None if invalid identifier
        
    Supports two lookup methods:
    1. Direct task_id lookup
    2. Numeric index lookup (for sequential access)
    """
    tid = identifier  # Initially assume identifier is a task_id
    info = task_index.get(tid)

    # If not found and identifier is numeric, try index-based lookup
    if info is None and identifier.isdigit():
        idx = int(identifier)
        if 0 <= idx < len(task_keys):
            tid = task_keys[idx]
            info = task_index.get(tid)

    if info is None:  # Still not found - invalid identifier
        return None

    # Handle JSONL format
    if info['type'] == 'jsonl':
        line_no = info['ptr']
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_no:
                    task = json.loads(line)
                    task['task_id'] = tid  # Ensure task_id is set
                    return task

    # Handle directory format
    task_path = info['ptr']
    meta_file = os.path.join(task_path, 'meta.json')
    
    # Load metadata
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    # Load all step files (excluding meta.json)
    step_files = sorted([
        f for f in os.listdir(task_path)
        if f.endswith('.json') and f != 'meta.json'
    ])
    
    steps = []
    for fname in step_files:
        with open(os.path.join(task_path, fname), 'r', encoding='utf-8') as sf:
            steps.append(json.load(sf))

    # Construct complete task object
    return {
        'instruction': meta.get('instruction', ''),
        'task_id': tid,
        'traj': steps,  # Trajectory/steps data
        'task_completed': meta.get('task_completed'),
        'alignment_score': meta.get('alignment_score'),
        'efficiency_score': meta.get('efficiency_score'),
        'reason': meta.get('reason', ''),
        'actual_task': meta.get('actual_task', ''),
        'natural_language_task': meta.get('natural_language_task', ''),
        'task_difficulty': meta.get('task_difficulty'),
        'redundant_step_count': meta.get('redundant_step_count')
    }


# Initialize the task index when module loads
init_task_index()

# ========== Flask Routes ==========

@app.route('/')
def index():
    """
    Serve the main index page.
    
    Returns:
        str: Rendered HTML template with total task count
    """
    # Pass total task count to template for display
    # Could also pass task_index.keys() if we want to show task ID list
    return render_template('index.html', total=total_tasks)


@app.route('/task/<task_id>')
def get_task(task_id):
    """
    API endpoint to retrieve a specific task by ID.
    
    Args:
        task_id (str): Task identifier (task_id or numeric index)
        
    Returns:
        JSON: Task data if found, error message if not found
    """
    task = load_task_by_id(task_id)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Invalid task ID'}), 404


# @app.route('/images/<path:filename>')
# def serve_image(filename):
#     """
#     Serve static image files from the configured image directory.
    
#     Args:
#         filename (str): Path to the image file relative to IMAGE_FOLDER
        
#     Returns:
#         File: The requested image file
#     """
#     return send_from_directory(IMAGE_FOLDER, filename)
# 在你的serve_image函数中，只需要添加一行打印语句：

@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve static image files from the configured image directory.
    """
    # 使用绝对路径避免路径拼接问题
    full_path = os.path.join(os.path.abspath(IMAGE_FOLDER), filename)
    print(f"Looking for image at: {full_path}")
    print(f"File exists: {os.path.exists(full_path)}")
    
    try:
        if not os.path.exists(full_path):
            print(f"ERROR: File does not exist at {full_path}")
            return "File not found", 404
        
        if not os.access(full_path, os.R_OK):
            print(f"ERROR: No read permission for {full_path}")
            return "Permission denied", 403
            
        print(f"SUCCESS: Serving {filename}")
        return send_file(full_path, mimetype='image/png')
        
    except Exception as e:
        print(f"ERROR serving image {filename}: {str(e)}")
        return f"Error: {str(e)}", 500

# ========== Application Entry Point ==========
if __name__ == '__main__':
    """
    Start the Flask development server.
    
    Server configuration:
    - Debug mode enabled for development
    - Listens on all interfaces (0.0.0.0)
    - Port configurable via command line argument
    """
    app.run(debug=True, host='0.0.0.0', port=args.port)