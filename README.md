# Gemma Refusal Direction Extraction

Extract the "refusal direction" from Gemma 7B IT using the method from [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) (Arditi et al., NeurIPS 2024).

Reference implementation: https://github.com/andyrdt/refusal_direction

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- ~16 GB RAM (the 7B model will be partially offloaded to disk on memory-constrained machines)
- A HuggingFace account with access to [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it) (gated model — you must accept the license)

## Setup


```bash
# Basic run
uv run extract_refusal_direction.py

# With a smaller batch size (for limited memory)
uv run extract_refusal_direction.py --batch_size 8

# With sanity-check generation at the end
uv run extract_refusal_direction.py --batch_size 8 --sanity

# Pass HF token inline
HF_TOKEN=hf_your_token_here uv run extract_refusal_direction.py --batch_size 8
```

### Without uv

```bash
pip install torch transformers tqdm accelerate
python extract_refusal_direction.py --batch_size 8
```
## Outputs

All outputs are saved to `./artifacts/` (or the path given by `--output_dir`):

| File | Description |
|------|-------------|
| `refusal_direction.pt` | The selected refusal direction vector (shape `[3072]`) |
| `direction_metadata.json` | The selected layer index and token position |
| `candidate_directions.pt` | All candidate directions (shape `[5, 28, 3072]`) |
| `all_direction_scores.json` | Scores for every candidate (KL, ablation, steering) |
