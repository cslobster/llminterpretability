# LLM Refusal Direction Extraction

Extract the "refusal direction" from language models using the method from [Refusal in Language Models Is Mediated by a Single Direction](https://proceedings.neurips.cc/paper_files/paper/2024/file/f545448535dfde4f9786555403ab7c49-Paper-Conference.pdf) (Arditi et al., NeurIPS 2024).

Two pipelines are provided:
1. **Text-only** (`extract_refusal_direction.py`) — Gemma 7B IT with text-only harmful/harmless prompts
2. **Vision-Language** (`extract_refusal_direction_vlm.py`) — PaliGemma 2 with multimodal (image+text) inputs from [MM-SafetyBench](https://huggingface.co/datasets/PKU-Alignment/MM-SafetyBench) and [COCO](https://huggingface.co/datasets/detection-datasets/coco)

Reference implementation: https://github.com/andyrdt/refusal_direction

## Prerequisites

- Python 3.10+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended) or pip
- ~16 GB RAM (the 7B text model will be partially offloaded to disk on memory-constrained machines)
- A HuggingFace account with access to [google/gemma-7b-it](https://huggingface.co/google/gemma-7b-it) (gated model — you must accept the license)

## Usage

### Text-only (Gemma 7B IT)

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

### Vision-Language (PaliGemma 2)

```bash
# Full run with PaliGemma 2 10B (requires >32 GB RAM or CUDA GPU)
uv run extract_refusal_direction_vlm.py

# Run with 3B model (fits on 32 GB Apple Silicon via MPS)
uv run extract_refusal_direction_vlm.py --model_id google/paligemma2-3b-mix-448

# Reduced samples for faster iteration
VLM_N_TRAIN=32 VLM_N_VAL=8 uv run extract_refusal_direction_vlm.py \
  --batch_size 2 --skip_filter --model_id google/paligemma2-3b-mix-448
```

### Without uv

```bash
pip install torch transformers tqdm accelerate datasets Pillow
python extract_refusal_direction.py --batch_size 8
python extract_refusal_direction_vlm.py --batch_size 2 --model_id google/paligemma2-3b-mix-448
```

## Outputs

### Text-only — `./artifacts/`

| File | Description |
|------|-------------|
| `refusal_direction.pt` | The selected refusal direction vector (shape `[3072]`) |
| `direction_metadata.json` | The selected layer index and token position |
| `candidate_directions.pt` | All candidate directions (shape `[5, 28, 3072]`) |
| `all_direction_scores.json` | Scores for every candidate (KL, ablation, steering) |

### Vision-Language — `./artifacts/vlm/`

| File | Description |
|------|-------------|
| `refusal_direction_vlm.pt` | The selected refusal direction vector (shape `[d_model]`) |
| `direction_metadata_vlm.json` | The selected position, layer, and model info |
| `candidate_directions_vlm.pt` | All candidate directions (shape `[3, n_layers, d_model]`) |
| `all_direction_scores_vlm.json` | Scores for every candidate (KL, ablation, steering) |

## Results

### Text-only: Gemma 7B IT

Run on a MacBook Pro with 32 GB memory (Apple Silicon, MPS backend).

Best direction found at **position=-1, layer=14** (out of 28 layers):

| Metric | With direction | Baseline |
|--------|---------------|----------|
| Ablation refusal score (harmful prompts) | -11.71 | 7.71 |
| Steering refusal score (harmless prompts) | 7.14 | -14.37 |
| KL divergence (harmless) | 0.079 | — |

Ablating this single direction from all layers flips the model from refusing harmful prompts (score +7.71) to complying (score -11.71). Conversely, adding the direction to harmless prompts induces refusal (score jumps from -14.37 to +7.14). The low KL divergence (0.079) confirms the intervention is surgical and does not degrade general model capabilities.

### Vision-Language: PaliGemma 2 3B

Run on the same MacBook Pro with `google/paligemma2-3b-mix-448` (pipeline validation; 10B model requires >32 GB RAM).

Dataset: **MM-SafetyBench** (harmful image+text pairs across 13 categories) vs **COCO** (harmless natural images with benign questions). N_TRAIN=32, N_VAL=8, `--skip_filter`.

Best direction found at **position=-1, layer=7** (out of 26 layers, hidden dim 2304):

| Metric | With direction | Baseline |
|--------|---------------|----------|
| Ablation refusal score (harmful prompts) | -5.50 | -4.81 |
| Steering refusal score (harmless prompts) | -7.99 | -10.65 |
| KL divergence (harmless) | 0.100 | — |

PaliGemma 2 3B is not heavily safety-tuned (baseline scores are negative, meaning it rarely refuses). Despite this, the extracted direction still captures refusal-related signal: ablation pushes the score further negative (-4.81 → -5.50), and steering significantly increases the refusal tendency on harmless inputs (-10.65 → -7.99). The KL divergence (0.100) shows the intervention is minimally destructive.

**Key optimizations for VLM pipeline**: KL-based pre-filtering reduces the candidate evaluation from 78 full model passes to 9, cutting Step 5 runtime from ~7 hours to ~22 minutes on MPS.
