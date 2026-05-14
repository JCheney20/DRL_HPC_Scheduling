import os
import pandas as pd
import json
import hashlib
from datetime import datetime
from pathlib import Path
from utils import ArgumentParserWithDefaults

def parse_args():
    parser = ArgumentParserWithDefaults(description='Trace Splitter')
    parser.add_argument('--source', '--src',
                        default="physical_job",
                        dest='source',
                        metavar='SRC',
                        choices = ["physical_job","deeplearn_job"],
                        help='Raw file to be split relative to data/',
                        type=str)
    parser.add_argument('--ratio',
                        default=0.7,
                        dest='ratio',
                        metavar='RATIO',
                        help='Ratio of dev:train',
                        type=float)
    parser.add_argument('--out-dir',
                        default='data/splits/',
                        dest='out_dir',
                        metavar='OUT_DIR',
                        help='Output directory of splits',
                        type=str)

    args = parser.parse_args()
    print(args)
    return args.source, \
           args.ratio, \
           args.out_dir 


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

if __name__ == '__main__':
    source, ratio, out_dir = parse_args() 
    src_path = f"data/{source}.csv"
    ratio_num = int(ratio*100)
    train_file = f"{source}_dev{ratio_num}.tsv"
    test_file = f"{source}_holdout{100-ratio_num}.tsv"
    dev_path = Path(out_dir) / train_file 
    holdout_path = Path(out_dir) / test_file 
    logs_dir = Path(out_dir) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Validate input
    if not os.path.exists(src_path):
        raise ValueError("Source file does not exist")
    
    assert 0 < ratio < 1, "Ratio must be between 0 and 1"

    headers = pd.read_csv(src_path, sep="\t", low_memory=False, nrows=0).columns
    assert "Submit" in headers, "Raw File does not have submit column"
    df = pd.read_csv(src_path, sep="\t", low_memory=False)

    # Sort & Split Dataframe
    df = df.sort_values(by="Submit", kind="mergesort")

    split_index = int(df.shape[0]*ratio)

    chunk = df.iloc[:split_index,:]
    remainder = df.iloc[split_index:,:]


    #Output to Split Files
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    chunk.to_csv(dev_path, index=False, sep="\t")
    remainder.to_csv(holdout_path, index=False, sep="\t")

    # Meta Data JSON file
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    split_id = f"{source}_r{ratio_num}"

    metadata = {
        "timestamp": timestamp,
        "split_id": split_id,
        "source_path": src_path,
        "source_sha256": sha256_file(src_path),
        "ratio": ratio,
        "sort_key": "Submit",
        "stable_sort": True,
        "total_rows": int(df.shape[0]),
        "dev_rows": int(chunk.shape[0]),
        "holdout_rows": int(remainder.shape[0]),
        "dev_path": str(dev_path),
        "holdout_path": str(holdout_path),
    }

    with open(logs_dir / f"{split_id}.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    #stdout
    out=("="*10)+f"""\nFile Split Metadata\nsource:\t\t\t{metadata.get("source_path")}\ntotal_rows:\t\t\t{metadata.get("total_rows")}\ntrain/dev_rows:\t\t\t{metadata.get("dev_rows")}\ntest/holdout_rows:\t\t\t{metadata.get("holdout_rows")}\nsplit_ratio:\t\t\t{metadata.get("ratio")}\nsplit_id:\t\t\t{metadata.get("split_id")}\ntrain/dev_path:\t\t\t{metadata.get("dev_path")}\ntest/holdout_path:\t\t\t{metadata.get("holdout_path")}\nmetadata_path:\t\t\t{logs_dir}/{split_id}.json\n"""+("="*10)+"\n\nSplit Complete."

    print(out)
