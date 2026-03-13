# The Double Barrier: Why Vision-Language Models Fail to Refuse Unsafe Images

Mechanistic interpretability analysis of cross-modal refusal in vision-language models. We identify the *double barrier* — a write-side gap and a read-side gap — that explains why VLMs refuse unsafe text but comply with unsafe images.

## Models

| Model | Architecture | LM Layers | Hidden Dim |
|-------|-------------|-----------|------------|
| **Qwen3.5-9B** | SigLIP + Hybrid Gated DeltaNet/Attention | 32 | 4096 |
| **InternVL2-8B** | InternViT-300M + InternLM2-7B | 32 | 4096 |

## Dataset

[HoliSafe-Bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench) — multimodal safety benchmark with five categories:

| Category | Image | Query | Expected |
|----------|-------|-------|----------|
| SSS | Safe | Safe | Comply |
| USU | Unsafe | Safe | Refuse |
| SUU | Safe | Unsafe | Refuse |
| UUU | Unsafe | Unsafe | Refuse |
| SSU | Safe | Safe (unsafe context) | Refuse |

## Experiments

| # | Experiment | Script | Description |
|---|-----------|--------|-------------|
| 1 | Refusal Rate | `measure_refusal_rate_qwen_holisafe_modal.py` | Baseline refusal rates across categories |
| 2 | Direction Geometry | `refusal_direction_geometry_qwen_modal.py` | Cosine similarity between text/vision refusal directions |
| 2b | Position Geometry | `position_refusal_geometry_qwen_modal.py` | Refusal directions at image vs text token positions |
| 3 | Semantic Proximity | `semantic_proximity_modal.py` | Whether text queries can rescue weak visual safety signal |
| 4+5 | Projection + Ablation | `ablation_projection_qwen_modal.py` | Projection strength analysis and rank-1 ablation |
| 5b | Subspace Ablation | `subspace_ablation_qwen_modal.py` | PCA-based multi-component ablation (negative control) |
| 6 | Steering Injection | `steering_injection_qwen_modal.py` | Position-selective refusal direction injection |
| All | InternVL2 | `internvl2_experiments_modal.py` | Experiments 1, 2, 4, 5, 6 for InternVL2-8B |

## Setup

### Prerequisites

- Python 3.11+
- [Modal](https://modal.com/) account (experiments run on A100 GPUs)
- HuggingFace token with access to gated models
- Gemini API key (for LLM judge)

### Installation

```bash
pip install modal
modal setup

# Store secrets
modal secret create huggingface HF_TOKEN=<your-token>
```

### Running Experiments

All experiments run on Modal with A100 GPUs. Use `--detach` for long runs:

```bash
# Exp 1: Refusal rates
modal run measure_refusal_rate_qwen_holisafe_modal.py

# Exp 2: Direction geometry
modal run refusal_direction_geometry_qwen_modal.py

# Exp 2b: Position-specific geometry (new)
modal run --detach position_refusal_geometry_qwen_modal.py

# Exp 3: Semantic proximity (requires query generation first)
GEMINI_API_KEY=... python generate_proximity_queries.py
GEMINI_API_KEY=... modal run semantic_proximity_modal.py

# Exp 4+5: Projection + ablation
modal run ablation_projection_qwen_modal.py

# Exp 5b: Subspace ablation
GEMINI_API_KEY=... modal run --detach subspace_ablation_qwen_modal.py

# Exp 6: Steering injection
modal run --detach steering_injection_qwen_modal.py

# InternVL2: All experiments
modal run --detach internvl2_experiments_modal.py
```

### Gemini Judge (post-hoc)

Some experiments use heuristic refusal detection on Modal, then Gemini as a more accurate judge locally:

```bash
GEMINI_API_KEY=... python judge_refusal_gemini.py
GEMINI_API_KEY=... python judge_ablation_gemini.py
GEMINI_API_KEY=... python judge_internvl2_gemini.py
```

### Data Download

```bash
# Download 100 samples per HoliSafe category (for local inspection)
python download_holisafe_100.py
```

## File Overview

### Experiment Scripts

| File | Purpose |
|------|---------|
| `measure_refusal_rate_qwen_holisafe_modal.py` | Qwen refusal rate measurement |
| `refusal_direction_geometry_qwen_modal.py` | Refusal direction cosine similarity + linear probes |
| `position_refusal_geometry_qwen_modal.py` | Position-specific (image/text/all/last) refusal geometry |
| `ablation_projection_qwen_modal.py` | Projection strength + rank-1 ablation |
| `ablation_dose_response_qwen_modal.py` | Dose-response ablation sweep |
| `subspace_ablation_qwen_modal.py` | PCA subspace ablation (negative control) |
| `steering_injection_qwen_modal.py` | Position-selective steering vector injection |
| `semantic_proximity_modal.py` | Semantic proximity curve |
| `internvl2_experiments_modal.py` | All InternVL2 experiments in one script |

### Utilities

| File | Purpose |
|------|---------|
| `download_holisafe_100.py` | Download 100 samples per category |
| `download_holisafe_samples.py` | Download sample images |
| `generate_proximity_queries.py` | Generate graded queries using Gemini Vision |
| `judge_refusal_gemini.py` | Gemini Flash refusal judge |
| `judge_ablation_gemini.py` | Gemini judge for ablation results |
| `judge_internvl2_gemini.py` | Gemini judge for InternVL2 results |
| `judge_refusal_qwen_modal.py` | Qwen self-judge (on Modal) |

### Legacy / Early Exploration

| File | Purpose |
|------|---------|
| `extract_refusal_direction_holisafe.py` | Local refusal direction extraction (PaliGemma) |
| `extract_refusal_direction_holisafe_modal.py` | Modal version (PaliGemma) |
| `extract_refusal_direction_gemma2b_modal.py` | Gemma-2B refusal direction extraction |
| `extract_refusal_direction_vlm.py` | VLM pipeline for PaliGemma 2 |
| `extract_refusal_direction.py` | Text-only pipeline for Gemma 7B IT |
| `extract_refusal_rate_qwen_modal.py` | Early Qwen refusal rate extraction |

## Key Results

**Refusal rate asymmetry**: Both models refuse text-triggered harm (SUU: 66-80%) but not vision-triggered harm (USU: 6%).

**Shared refusal direction**: Text and vision refusal directions converge to cosine similarity 0.73-0.84 in late layers.

**Write-side gap**: Vision encoder writes only 43-73% of text signal magnitude along the refusal direction.

**Read-side gap**: Steering injection at text positions drives refusal to 88-100%; at image positions, only 8-11%.

**Rank-1 mechanism**: PCA subspace ablation *increases* refusal — the mechanism is specifically the mean-difference direction, not a broader subspace.
