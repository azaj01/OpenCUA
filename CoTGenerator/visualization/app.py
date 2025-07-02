from flask import Flask, render_template, send_from_directory, jsonify
import json, os, glob, argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

lock = threading.Lock()   
def _scan_subdir(subdir:str):
    """扫描单个子目录，符合条件则返回 (tid, task_path)"""
    task_path = os.path.join(DATA_PATH, subdir)
    meta_file = os.path.join(task_path, 'meta.json')
    if not (os.path.isdir(task_path) and os.path.exists(meta_file)):
        return None
    step_files = [f for f in glob.glob(os.path.join(task_path, '*.json'))
                  if os.path.basename(f) != 'meta.json']
    if not step_files:
        return None
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)
    tid = str(meta.get('task_id', subdir))
    return tid, task_path


app = Flask(__name__)
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='./OpenCUA-Code/CoTGenerator/example/output')
parser.add_argument('--image_folder', type=str,
                    default="./OpenCUA-Code/CoTGenerator/example/images")
parser.add_argument('--port', type=int, default=5000)
args = parser.parse_args()

DATA_PATH     = args.data_path
IMAGE_FOLDER  = args.image_folder

task_index = {}      # tid  -> {'type': ..., 'ptr': ...}
task_keys  = []      # 保持顺序的 tid 列表
total_tasks = 0



def init_task_index():
    """多线程建立 task_id → 位置 的映射"""
    global task_index, total_tasks

    if os.path.isfile(DATA_PATH):
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f):
                data = json.loads(line)
                tid  = str(data.get('task_id', line_no))
                task_index[tid] = {'type': 'jsonl', 'ptr': line_no}
                task_keys.append(tid)                          # ←★ 新增
    else:                                              # ---------- 目录 ----------
        subdirs = sorted(os.listdir(DATA_PATH))
        with ThreadPoolExecutor(max_workers=128) as pool:
            futures = [pool.submit(_scan_subdir, sd) for sd in subdirs]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                result = fut.result()
                if result:
                    tid, task_path = result
                    with lock:
                        task_index[tid] = {'type': 'dir', 'ptr': task_path}
                        task_keys.append(tid)                  

    total_tasks = len(task_index)
    print(f'索引完成: 找到 {total_tasks} 个任务')


def load_task_by_id(identifier: str):
    """
    既支持字符串 task_id，也支持数字顺序索引。
    identifier: task_id 字符串或 '0'、'1' 这样的数字
    """
    tid  = identifier                 # 先假设就是 tid
    info = task_index.get(tid)

    # 如果没找到且 identifier 是数字，则按顺序索引再找
    if info is None and identifier.isdigit():
        idx = int(identifier)
        if 0 <= idx < len(task_keys):
            tid  = task_keys[idx]
            info = task_index.get(tid)

    if info is None:                  # 还没找到 → 无效
        return None

    # ---------- JSONL ----------
    if info['type'] == 'jsonl':
        line_no = info['ptr']
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == line_no:
                    task = json.loads(line)
                    task['task_id'] = tid
                    return task

    # ---------- 目录 ----------
    task_path = info['ptr']
    meta_file = os.path.join(task_path, 'meta.json')
    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    step_files = sorted(
        [f for f in os.listdir(task_path)
         if f.endswith('.json') and f != 'meta.json']
    )
    steps = []
    for fname in step_files:
        with open(os.path.join(task_path, fname), 'r', encoding='utf-8') as sf:
            steps.append(json.load(sf))

    return {
        'instruction'          : meta.get('instruction', ''),
        'task_id'              : tid,
        'traj'                 : steps,
        'task_completed'       : meta.get('task_completed'),
        'alignment_score'      : meta.get('alignment_score'),
        'efficiency_score'     : meta.get('efficiency_score'),
        'reason'               : meta.get('reason', ''),
        'actual_task'          : meta.get('actual_task', ''),
        'natural_language_task': meta.get('natural_language_task', ''),
        'task_difficulty'      : meta.get('task_difficulty'),
        'redundant_step_count' : meta.get('redundant_step_count')
    }


# ---------- 初始化 ----------
init_task_index()

# ---------- 路由 ----------
@app.route('/')
def index():
    # 首页只需要任务总数；如要显示 id 列表也可把 task_index.keys() 传给模板
    return render_template('index.html', total=total_tasks)

@app.route('/task/<task_id>')      # <int:...> 改成字符串
def get_task(task_id):
    task = load_task_by_id(task_id)
    if task:
        return jsonify(task)
    return jsonify({'error': 'Invalid task ID'}), 404

@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=args.port)
