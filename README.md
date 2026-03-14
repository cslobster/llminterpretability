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
modal run <experiment_script>.py
```

Set `GEMINI_API_KEY` environment variable for scripts that use the Gemini judge.
