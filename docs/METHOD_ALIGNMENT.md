# Manuscript-to-code correspondence

Reference: *Hyperbolic Retrieval for Hierarchical Evidence Completion in Multimodal Emotion Recognition*. The method appears on PDF pages 6–18 and training settings on page 20. The reference PDF SHA-256 is `56131182f0c2802c869e270d1c9a29f4cc2fdcbdf22ff30eee55f5c9c8d6cd24`.

| Manuscript component | Implementation | Operational definition |
|---|---|---|
| Eq. 2: exponential/logarithmic maps | `hrec/geometry.py` | General curvature, continuous origin behavior, numerical ball projection |
| Eq. 3: coordinates and paired evidence | `HyperbolicRetriever.encode`, `build_memory` | Modality-specific projections and masked pooling of frozen encoded sequences |
| Eqs. 4–7: eligible modalities and dynamic query | `query`, `candidate_distances`, `retrieve` | A/V excludes text when available; text-only fallback; identical query-specific mixture weights for query and candidates |
| Eqs. 8–9: geodesic distance | `PoincareBall.mobius_add`, `distance` | Retrieval geometry in float32 |
| Eq. 10: intensity harmonization | `sentiment_level`, `annotation_path` | MOSEI scale 3 and SIMS-V2 scale 1; seven intensity levels |
| Eq. 11: hierarchy supervision | `EmotionTree.candidates`, `pretraining_loss` | 96 distinct candidates in three groups; soft target distribution and cross-entropy |
| Eqs. 12–14: distance-aware completion | `EvidenceCompletion.forward` | Separate A/V projections; log distance weights added to content logits |
| Eq. 15: residual completion | `EvidenceCompletion.gate` | Scalar gate from current/retrieved summaries and availability |
| Eqs. 16–19: descriptor and balancing | `HREC.forward` | Weighted mean distance, normalized entropy, availability; softmax modality weights; one aligned token |
| Eq. 20: textual evidence | `FrozenLanguageModel.format_inputs` | Caption, emotion, distance; prompt, multimodal token, retrieved text |
| Eqs. 21–22: prediction and loss | `FrozenLanguageModel.forward`, `task_loss` | Last non-padding state; classification cross-entropy or regression SmoothL1 |
| Algorithm 1: two-stage training | `train_retriever`, `build_memory`, `train_task` | Rebuilt frozen memory; frozen retriever/LLM during task training |
| Retrieval analysis | `retrieval_metrics` | Labels accessed after ranking; MTD@k and full-memory ideal H-nDCG@k |

## Backbone-specific implementation

All three variants use the shared HREC equations and optimizer. Their source directories contain the transformer implementations actually loaded for training:

- Qwen selects the local `QWenLMHeadModel` transformer body, with explicit precision flags and optional FlashAttention disabled.
- ChatGLM accepts differentiable batch-first embeddings, converts to its native sequence-first layout, and preserves causal masks and rotary positions. Its tokenizer supports checkpoint save/reload with fixed special tokens. Quantized and independently prefix-tuned checkpoints are excluded from the supported configuration.
- Llama uses the bundled Transformers 4.40.2 implementation with explicit position IDs and eager attention.

The actual HREC and transformer classes are recorded in `training_report.json`. Source hashes and modifications are recorded in `BACKBONE_SOURCE_MANIFEST.json`.

## Explicit implementation choices

The following choices make details not fully specified in the manuscript executable:

1. Pre-extracted A/V features and frozen backbone word embeddings enter single-layer LSTM encoders with 128 hidden units and projection to 256 dimensions. Summaries use masked means. Exact raw extractors must be specified with the data.
2. Completion uses single-head attention of dimension 128 and output dimension 256. Retrieval, completion, and balancing gates use two-layer ReLU networks with 128 hidden units. Attention dropout is 0.1.
3. Half-integer intensity quantization rounds away from zero. A fixed synonym map covers supplied common English and Chinese labels; unknown fine-grained labels require reviewed paths. Equivalent basic labels share one node.
4. Surprise follows the training-protocol text's undetermined polarity assignment, rather than the positive assignment in the illustrative figure.
5. Fully observed training records sample the five observation patterns uniformly. Naturally incomplete records use patterns retaining an observation. Feature-level Gaussian noise is configurable and disabled by default because its distribution and strength are unspecified in the manuscript.
6. Effective batch size 32 is implemented with micro-batches of four. Final partial batches use their actual sample count. Both stages use AdamW, linear warmup/decay, and gradient clipping; validation loss selects the task checkpoint.
7. Prompt and evidence token budgets are explicit. The retrieval budget is divided among neighbors; label and distance fields are retained before caption truncation. Insufficient budgets raise an error.
8. The discriminative task head reads the final valid hidden state. Pretrained generation heads and utilities retained in the backbone source are not used for the HREC objective.

## Reproduction scope

The release provides the method and an executable training/evaluation procedure. It does not include the original raw-feature extraction pipeline, complete external feature memory, or completed benchmark runs. Software validation with synthetic features, including validation with pretrained Qwen weights, establishes execution and gradient behavior within its stated scope. See [../VALIDATION.md](../VALIDATION.md) and [DATA.md](DATA.md).
