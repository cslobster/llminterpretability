# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "datasets",
#     "Pillow",
# ]
# ///
"""
Download 100 samples per category from HoliSafe-Bench and save to data/holisafe_100/.

Categories: SSS, USU, SUU, UUU, SSU.
Each category gets a subfolder with images and a metadata.json.

Usage:
    uv run download_holisafe_100.py
"""

import json
import os
import random

from datasets import load_dataset

HARMFUL_TYPES = ["USU", "SUU", "UUU", "SSU"]
HARMLESS_TYPE = "SSS"
ALL_TYPES = [HARMLESS_TYPE] + HARMFUL_TYPES
N_SAMPLES = 100
OUTPUT_DIR = "data/holisafe_100"
SEED = 42


def main():
    token = None
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        token = open(token_path).read().strip()

    print("Loading HoliSafe-Bench dataset ...")
    ds = load_dataset("etri-vilab/holisafe-bench", split="test", token=token)

    rng = random.Random(SEED)

    for type_code in ALL_TYPES:
        filtered = ds.filter(lambda x, tc=type_code: x["type"] == tc)
        valid_indices = [
            i for i, row in enumerate(filtered)
            if row.get("image") is not None and row.get("query", "")
        ]

        n = min(N_SAMPLES, len(valid_indices))
        if n < N_SAMPLES:
            print(f"  WARNING: {type_code} only has {n} valid samples (requested {N_SAMPLES})")
        selected = rng.sample(valid_indices, n)

        out_dir = os.path.join(OUTPUT_DIR, type_code)
        os.makedirs(out_dir, exist_ok=True)

        metadata = []
        for idx, i in enumerate(selected):
            row = filtered[i]
            img = row["image"].convert("RGB")
            query = row["query"]
            img_path = os.path.join(out_dir, f"{idx:02d}.png")
            img.save(img_path)
            metadata.append({"index": idx, "query": query, "image": f"{idx:02d}.png"})

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"  {type_code}: saved {len(metadata)} samples to {out_dir}/")

    print("Done.")


if __name__ == "__main__":
    main()
