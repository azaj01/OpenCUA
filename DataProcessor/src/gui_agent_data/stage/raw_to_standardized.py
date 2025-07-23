import argparse
import importlib
import os
import multiprocessing as mp
from pathlib import Path

import orjson as json
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str)
parser.add_argument("raw_file", type=str)
parser.add_argument("--num_samples", type=int, default=-1)
parser.add_argument("output_filename", type=str)
args = parser.parse_args()

# makedir
Path(args.output_filename).mkdir(parents=True, exist_ok=True)

convert_examples = importlib.import_module(
    f"gui_agent_data.datasets.{args.dataset}.raw_to_standardized"
).convert_examples

processed_episode_ids = set()
for item in os.listdir(args.output_filename):
    processed_episode_ids.add(item.split(".json")[0])

if args.raw_file.endswith(".json"):
    with open(args.raw_file, encoding="utf-8") as f:
        raw_examples = json.loads(f.read())

    if args.num_samples != -1:
        raw_examples = raw_examples[:args.num_samples]

    for raw_example in tqdm(raw_examples):
        # print(raw_example.keys())
        if raw_example['episode_id'] in processed_episode_ids:
            continue
        raw_example_list = [raw_example]
        converted_example = convert_examples(raw_example_list)[0]

        with open(f"{args.output_filename}/{converted_example.example_id}.json", "wb") as f:
            f.write(json.dumps(converted_example.dict()))

else:
    # check if the raw_file is a directory
    print(f"Processing {args.raw_file}")

    # add a threshold to the number of files to process
    if args.num_samples != -1:
        raw_files = list(Path(args.raw_file).glob("*.json"))[:args.num_samples]
    else:
        raw_files = list(Path(args.raw_file).glob("*.json"))

    for raw_file in tqdm(raw_files):
        # print("haha")
        episode_id = raw_file.stem
        if episode_id in processed_episode_ids:
            continue

        with open(raw_file, encoding="utf-8") as f:
            raw_example = json.loads(f.read())

        # if raw_example['episode_id'] in processed_episode_ids:
        #     continue
        raw_example_list = [raw_example]
        try:
            converted_examples = convert_examples(raw_example_list)
            if len(converted_examples) > 0:
                converted_example = converted_examples[0] 
            else:
                continue
        except Exception as e:
            raise ValueError(f"Error in {raw_file}: {e}")

        with open(f"{args.output_filename}/{converted_example.example_id}.json", "wb") as f:
            f.write(json.dumps(converted_example.dict()))
