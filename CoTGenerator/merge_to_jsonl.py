import json
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

def process_subdir(subdir):
    meta_file = subdir / "meta.json"
    if not meta_file.exists():
        return None

    try:
        with open(meta_file, "r") as f:
            meta_data = json.load(f)

        traj = []
        json_files = sorted(
            [f for f in subdir.glob("*.json") if f.name != "meta.json"],
            key=lambda x: int(x.stem) if x.stem.isdigit() else float("inf")
        )
        for jf in json_files:
            with open(jf, "r") as f:
                traj.append(json.load(f))

        meta_data["traj"] = traj
        return meta_data

    except Exception as e:
        print(f"Error processing {subdir}: {e}")
        return None

def main():
    argparser = argparse.ArgumentParser(description="Merge JSON files into a single JSONL file.")
    argparser.add_argument("--input_dir", type=str, default="./OpenCUA-Code/CoTGenerator/example/output", help="Input directory containing JSON files.")
    argparser.add_argument("--output_file", type=str, default="traj_with_cot_example.jsonl", help="Output path for the merged JSONL file.")
    args = argparser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = input_dir.parent / args.output_file

    subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"📂 Find {len(subdirs)} tasks...")

    output_lines = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_subdir, subdirs), total=len(subdirs)):
            if result is not None:
                output_lines.append(result)

    print(f"✅ Successfully processed {len(output_lines)} / {len(subdirs)}...")

    with open(output_path, "w") as f:
        for item in output_lines:
            json.dump(item, f)
            f.write("\n")

    print(f"🎉 Merge finished, save to {output_path}")

if __name__ == "__main__":
    main()
