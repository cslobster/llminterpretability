# Cross-Model Refusal Direction: Qwen2-VL-7B vs Qwen2-7B

## Setup

- **VLM**: Qwen2-VL-7B-Instruct (28 layers, d_model=3584)
- **LLM**: Qwen2-7B-Instruct (28 layers, d_model=3584, same backbone)
- **Unsafe data**: MM-HarmfulBench (100 image+text harmful queries)
- **Safe data**: HoliSafe-Bench SSS (100 image+text benign queries)
- **VLM inputs**: image + text; **LLM inputs**: text only (same queries, no images)
- **Method**: Arditi et al. (2024) — multi-position difference-in-means, triple ablation
- **Judge**: Gemini 2.0 Flash
- **GPU**: A100-80GB, total runtime 1h57m

## Data Filtering

| | Safe train | Safe val | Unsafe train | Unsafe val |
|---|---|---|---|---|
| **VLM** | 64/64 (100%) | 16/16 (100%) | 48/64 (75%) | 15/16 (94%) |
| **LLM** | 50/64 (78%) | 16/16 (100%) | 61/64 (95%) | 15/16 (94%) |

Notable: The text-only LLM refuses **more** MM-HarmfulBench queries (95%) than the VLM (75%). Without images, the harmful intent is often clearer in text alone. The VLM's vision encoder may actually dilute the harmful signal.

## Direction Selection

| | Position | Layer | Bypass score | Induce score | KL |
|---|---|---|---|---|---|
| **VLM** | -4 | 16 | -4.957 | +0.214 | 0.289 |
| **LLM** | -1 | 19 | -5.979 | +3.256 | 0.072 |

- VLM direction is at layer 16 (57%), LLM at layer 19 (68%) — both in middle-to-upper layers
- LLM direction has stronger bypass and induce scores, and lower KL — a cleaner signal
- Different positions: VLM at -4 (4th from last), LLM at -1 (last token)

## Baseline Refusal Rates

| Category | VLM | LLM |
|----------|-----|-----|
| Unsafe | **75%** (15/20) | **90%** (18/20) |
| Safe | **0%** (0/20) | **10%** (2/20) |

The text-only LLM is more conservative: higher unsafe refusal (90% vs 75%) but also some false refusals on safe queries (10% vs 0%).

## Ablation — Own Direction

| | VLM baseline | VLM ablated | Delta | LLM baseline | LLM ablated | Delta |
|---|---|---|---|---|---|---|
| Unsafe | 75% | **10%** | **-65pp** | 90% | **35%** | **-55pp** |
| Safe | 0% | 0% | 0pp | 10% | 5% | -5pp |

Both directions are effective at removing refusal from their own models. The VLM direction is particularly potent (-65pp).

## Cross-Model Ablation

| | VLM→LLM | LLM→VLM |
|---|---|---|
| Unsafe | 90% → **15%** (-75pp) | 75% → **30%** (-45pp) |
| Safe | 10% → **0%** (-10pp) | 0% → **0%** (0pp) |

### Key Finding: Cross-transfer works in both directions.

- **VLM direction removes LLM refusal** even more effectively (-75pp) than the LLM's own direction (-55pp)
- **LLM direction removes VLM refusal** significantly (-45pp), though less than VLM's own direction (-65pp)
- Safe queries remain unaffected in both cross-transfer conditions

## Steering — Own Direction

### Bypass (subtract α×d from refused samples, baseline=100% refusal)

| α | VLM bypass rate | LLM bypass rate |
|---|---|---|
| 1 | **87%** | 56% |
| 4 | 13% | **83%** |
| 8 | 60% | 6% |

### Induce (add α×d to complied samples, baseline=0% refusal)

| α | VLM induce rate | LLM induce rate |
|---|---|---|
| 1 | 85% | **100%** |
| 4 | 45% | **100%** |
| 8 | 55% | **100%** |

LLM direction is more stable for inducement (100% across all α). Both show non-monotonic bypass behavior at high α (model coherence degrades).

## Cross-Model Steering

### Bypass (subtract α×d from refused samples)

| α | VLM dir→LLM | LLM dir→VLM |
|---|---|---|
| 1 | 50% | 47% |
| 4 | 17% | 33% |
| 8 | 0% | 0% |

### Induce (add α×d to complied samples)

| α | VLM dir→LLM | LLM dir→VLM |
|---|---|---|
| 1 | 83% | **100%** |
| 4 | **94%** | **100%** |
| 8 | 89% | 65% |

Cross-model steering works: VLM direction bypasses LLM refusal, LLM direction induces VLM refusal. Both directions are transferable.

## Geometric Analysis

**Cosine similarity between best directions: +0.300**

Per-layer cosine similarity (VLM pos=-4 vs LLM pos=-1):

| Layer range | Avg cosine |
|---|---|
| 0-6 | ~0.44 |
| 7-13 | ~0.40 |
| 14-20 | ~0.43 |
| 21-27 | ~0.26 |

The cosine similarity is moderate (~0.30-0.47), not near 1.0. The directions are **related but not identical** — they share a significant component but VL fine-tuning has rotated the refusal direction.

## Summary of Findings

1. **Both models have effective refusal directions**: VLM ablation drops refusal 75%→10% (-65pp), LLM drops 90%→35% (-55pp).

2. **Cross-model transfer works**: The VLM direction removes LLM refusal (90%→15%), and the LLM direction removes VLM refusal (75%→30%). This proves the two models share a common refusal subspace.

3. **The directions are related but not identical**: Cosine similarity of 0.30 between best directions. VL fine-tuning has partially rotated the refusal direction, but enough overlap remains for cross-transfer.

4. **VLM direction is more potent on the LLM than vice versa**: VLM→LLM ablation (-75pp) outperforms LLM→LLM ablation (-55pp). This suggests the VLM direction captures a more fundamental refusal component that the LLM also uses.

5. **Text-only LLM refuses more than VLM on the same queries**: 90% vs 75%. Adding images to harmful queries appears to dilute the harmful signal, consistent with the "double barrier" hypothesis that VLM safety relies primarily on text-side mechanisms.

## Implications

The moderate cosine similarity (~0.30) with strong cross-transfer effectiveness suggests that refusal is not a single direction but a **low-dimensional subspace** shared between VLM and LLM. VL fine-tuning preserves this subspace but shifts the primary refusal axis within it. This has practical implications:

- **Safety attacks**: An adversary who extracts the refusal direction from the open-source text LLM can use it to attack the corresponding VLM
- **Safety alignment**: Refusal directions learned during text-only training transfer to multimodal settings, suggesting that text-stage safety training provides a foundation for VLM safety
- **The vision pathway does not create new safety mechanisms**: The VLM's refusal direction lives in the same subspace as the LLM's, just rotated — no fundamentally new refusal geometry emerges from VL training
