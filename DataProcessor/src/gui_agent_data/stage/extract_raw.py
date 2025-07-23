import argparse
import importlib
import json
import os
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Name of the dataset")
    parser.add_argument("sample_raw", type=str, help="Output file path for raw samples")
    parser.add_argument("--num_samples", "-n", type=int, default=-1,
                        help="Number of samples to extract. Default: -1 (all samples)")

    args = parser.parse_args()

    module = importlib.import_module(f"gui_agent_data.datasets.{args.dataset}.extract_raw")
    raw_examples = module.get_raw_examples(args.num_samples)

    os.makedirs(os.path.join("datasets", args.dataset), exist_ok=True)
    if args.sample_raw.endswith(".json"):
        with open(args.sample_raw, "w") as f:
            json.dump(list(raw_examples), f, indent=2)
    else:
        Path(args.sample_raw).mkdir(parents=True, exist_ok=True)
        fetched_raw = set()
        for file in os.listdir(args.sample_raw):
            fetched_raw.add(file.split(".json")[0])
        for i, raw_example in enumerate(raw_examples):
            episode_id = raw_example["episode_id"]
            if episode_id in fetched_raw:
                continue
            with open(os.path.join(args.sample_raw, f"{episode_id}.json"), "w") as f:
                json.dump(raw_example, f, indent=2)
        
