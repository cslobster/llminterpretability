# VLM Safety Interpretability

Mechanistic interpretability research on safety mechanisms in vision-language models. Work in progress — paper forthcoming.

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
# Example
modal run measure_refusal_rate_qwen_holisafe_modal.py

# Scripts requiring Gemini judge
GEMINI_API_KEY=... modal run semantic_proximity_modal.py

# Post-hoc Gemini judging (runs locally)
GEMINI_API_KEY=... python judge_refusal_gemini.py
```

### Data Download

```bash
# Download samples from HoliSafe-Bench for local inspection
python download_holisafe_100.py
```

## Dataset

[HoliSafe-Bench](https://huggingface.co/datasets/etri-vilab/holisafe-bench) — multimodal safety benchmark with controlled image-query safety pairings:

| Category | Image | Query | Expected |
|----------|-------|-------|----------|
| SSS | Safe | Safe | Comply |
| USU | Unsafe | Safe | Refuse |
| SUU | Safe | Unsafe | Refuse |
| UUU | Unsafe | Unsafe | Refuse |
| SSU | Safe | Safe (unsafe context) | Refuse |

## Experiments

| # | Experiment | Script |
|---|-----------|--------|
| 1 | Refusal Rate Measurement | `measure_refusal_rate_qwen_holisafe_modal.py` |
| 2 | Refusal Direction Geometry | `refusal_direction_geometry_qwen_modal.py` |
| 2b | Position-Specific Geometry | `position_refusal_geometry_qwen_modal.py` |
| 3 | Semantic Proximity Curve | `semantic_proximity_modal.py` |
| 4+5 | Projection + Ablation | `ablation_projection_qwen_modal.py` |
| 5b | Subspace Ablation | `subspace_ablation_qwen_modal.py` |
| 6 | Steering Injection | `steering_injection_qwen_modal.py` |
| All | InternVL2 Experiments | `internvl2_experiments_modal.py` |

## File Overview

### Experiment Scripts

| File | Purpose |
|------|---------|
| `measure_refusal_rate_qwen_holisafe_modal.py` | Qwen refusal rate measurement |
| `refusal_direction_geometry_qwen_modal.py` | Refusal direction cosine similarity + linear probes |
| `position_refusal_geometry_qwen_modal.py` | Position-specific (image/text/all/last) refusal geometry |
| `ablation_projection_qwen_modal.py` | Projection strength + rank-1 ablation |
| `ablation_dose_response_qwen_modal.py` | Dose-response ablation sweep |
| `subspace_ablation_qwen_modal.py` | PCA subspace ablation |
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
