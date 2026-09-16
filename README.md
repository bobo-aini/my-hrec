# HREC

**Hyperbolic Retrieval for Hierarchical Evidence Completion in Multimodal Emotion Recognition**

This repository provides the HREC implementation accompanying the manuscript. It includes hierarchical retrieval pretraining, a frozen multimodal memory, geodesic-guided evidence completion, distance-conditioned modality balancing, and prediction with a frozen language model. Emotion classification and sentiment regression are supported.

Three model variants are provided. Each directory contains the transformer and tokenizer source, a model-specific HREC integration, an executable entry point, and four dataset configurations. The HREC retrieval, completion, and optimization modules are shared across variants.

| Variant | Transformer and tokenizer source | HREC integration | Text embedding dimension |
|---|---|---|---:|
| Qwen-1.8B | [HREC_Qwen_1_8B/backbone](HREC_Qwen_1_8B/backbone) | [HRECQwen](HREC_Qwen_1_8B/model.py) | 2048 |
| ChatGLM3-6B | [HREC_ChatGLM3_6B/backbone](HREC_ChatGLM3_6B/backbone) | [HRECChatGLM](HREC_ChatGLM3_6B/model.py) | 4096 |
| Llama2-7B | [HREC_Llama2_7B/backbone](HREC_Llama2_7B/backbone) | [HRECLlama](HREC_Llama2_7B/model.py) | 4096 |

The archive contains source code and configurations. Dataset assets, pretrained language-model weights, and benchmark-trained HREC checkpoints are not included. The recorded software checks and their scope are documented in [VALIDATION.md](VALIDATION.md); they do not constitute a reproduction of the manuscript's benchmark results.

## Repository organization

```text
HREC/
|-- HREC_Qwen_1_8B/       # Qwen source, HREC integration, configs, run.py
|-- HREC_ChatGLM3_6B/     # ChatGLM source, HREC integration, configs, run.py
|-- HREC_Llama2_7B/       # Llama source, HREC integration, configs, run.py
|-- hrec/                # Shared method, data interfaces, training, metrics
|-- configs/             # Equivalent flat layout of the 12 configurations
|-- scripts/             # Pipeline, repeated runs, integration checks, Slurm
|-- tests/               # Mechanism and backbone implementation tests
|-- docs/                # Data specification and manuscript-to-code mapping
`-- validation/          # Recorded software validation results
```

## Environment

The reference environment uses Python 3.10.16, PyTorch 2.6.0, CUDA 12.4, and Transformers 4.40.2. GPU validation was performed on an NVIDIA H100 80 GB. The bundled source adaptations use the pinned Transformers version.

From the extracted `HREC` directory:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-tested.txt
python -m pip install -e . --no-deps
```

On Windows, use `py -3.10 -m venv .venv` and `.\.venv\Scripts\Activate.ps1`. Install a CUDA-enabled PyTorch distribution appropriate for the local driver when running on a GPU. `requirements.txt` specifies supported dependency ranges; `requirements-tested.txt` records the versions used for validation.

## Installation and implementation checks

The following commands require no datasets or pretrained weights:

```bash
python -m unittest discover -s tests -v
python -m hrec demo --directory runs/check_classification --task classification
python -m hrec demo --directory runs/check_regression --task regression
python scripts/check_local_backbones.py --directory runs/check_local_backbones --output runs/local_backbones.json
```

The `demo` commands exercise the complete two-stage pipeline using synthetic features and a small frozen test model. The unit tests exercise the bundled Qwen, ChatGLM, and Llama architectures at reduced dimensions. The final command additionally checks local tokenizer/checkpoint loading and two-stage training through the separate ChatGLM and Llama entry points, using randomly initialized checkpoints. Use a new output directory for each run.

## Pretrained backbones

Obtain the corresponding checkpoint and tokenizer assets from the provider, and set `llm_path` in the selected configuration. Default locations are:

| Model | Provider | Local directory |
|---|---|---|
| Qwen-1.8B | [Qwen/Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B) | `pretrained/Qwen-1_8B` |
| ChatGLM3-6B | [zai-org/chatglm3-6b](https://huggingface.co/zai-org/chatglm3-6b) | `pretrained/chatglm3-6b` |
| Llama2-7B | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) | `pretrained/Llama-2-7b-hf` |

Required assets include `config.json`, all weight shards and their index, and tokenizer files (`qwen.tiktoken` or `tokenizer.model`, with the associated tokenizer configuration). Access and use are subject to provider terms. Record the checkpoint revision used for each experiment.

The three loaders instantiate the model classes distributed in this archive. Execution does not require fetching Python model files at runtime. `local_files_only` defaults to `true`. Language-model parameters remain frozen during task training; HREC task checkpoints store the trained task parameters separately from the pretrained backbone.

## Data preparation

The implementation consumes pre-extracted audio and visual sequences, observed text, and reviewed hierarchical annotations for the external training memory. Text embeddings are extracted from the selected frozen backbone. [docs/DATA.md](docs/DATA.md) specifies the schemas, supported benchmark import formats, and preparation commands.

Each experiment requires four JSONL manifests:

```text
data/<backbone>/<dataset>/train.jsonl
data/<backbone>/<dataset>/valid.jsonl
data/<backbone>/<dataset>/test.jsonl
data/<backbone>/<dataset>/memory.jsonl
```

The memory contains external training records with paired audio, visual, and text features, captions, and emotion annotations. Query and memory features must share the same extraction and preprocessing configuration. The standard 96-candidate retrieval objective requires at least 96 eligible memory entries. Dataset splits and source-video or conversation identifiers are checked before training.

Set `feature_dims`, `llm_path`, the four manifest paths, and `run_dir` in the selected JSON configuration. All commands below are executed from the repository root; configuration paths are resolved relative to that directory. Raw audio/video extraction is outside the released implementation and must be specified with the supplied feature data.

## Training

Each variant has an independent entry point. The following commands run data validation, retrieval pretraining, memory construction, task training, and test evaluation for MELD:

```bash
python HREC_Qwen_1_8B/run.py pipeline --config HREC_Qwen_1_8B/configs/meld.json
python HREC_ChatGLM3_6B/run.py pipeline --config HREC_ChatGLM3_6B/configs/meld.json
python HREC_Llama2_7B/run.py pipeline --config HREC_Llama2_7B/configs/meld.json
```

Execute the command for the selected backbone. Each model directory also provides `cherma.json`, `mosei.json`, and `simsv2.json`. MOSEI and SIMS-V2 use regression; MELD and CHERMA use classification. The flat `configs/hrec_<backbone>_<dataset>.json` files are equivalent presets.

Stages may also be executed individually:

```bash
python HREC_Qwen_1_8B/run.py audit --config HREC_Qwen_1_8B/configs/meld.json
python HREC_Qwen_1_8B/run.py pretrain --config HREC_Qwen_1_8B/configs/meld.json
python HREC_Qwen_1_8B/run.py build-memory --config HREC_Qwen_1_8B/configs/meld.json
python HREC_Qwen_1_8B/run.py train --config HREC_Qwen_1_8B/configs/meld.json
python HREC_Qwen_1_8B/run.py evaluate --config HREC_Qwen_1_8B/configs/meld.json --split test --pattern all
```

Stage 1 trains modality encoders, hyperbolic projections, and the retrieval gate for five epochs. The memory is then rebuilt using the frozen retriever. Stage 2 trains evidence completion, modality balancing, alignment, and the prediction head for 20 epochs. Gradients propagate through the frozen language model to the trainable HREC modules.

| Parameter | Default |
|---|---:|
| Hyperbolic dimension / curvature | 128 / 1.0 |
| Encoded feature dimension | 256 |
| Retrieval candidates / retrieved neighbors | 96 / 10 |
| Hierarchy / retrieval / completion temperatures | 0.5 / 0.5 / 0.1 |
| Effective batch size / micro-batch size | 32 / 4 |
| AdamW learning rate / weight decay | 0.0001 / 0.01 |
| Warmup fraction / gradient clipping norm | 0.1 / 1.0 |
| Dropout | 0.1 |

The task checkpoint with the lowest validation loss is retained. Each backbone/dataset/seed combination has its own retriever, memory, and task checkpoint. Existing completed checkpoints are not overwritten. Completed stages can be reused by subsequent commands; resuming an interrupted optimizer state within a stage is not implemented.

## Evaluation and repeated runs

Supported observation patterns are `all`, `no_audio`, `no_visual`, `no_text`, and `text_only`:

```bash
python HREC_Qwen_1_8B/run.py evaluate --config HREC_Qwen_1_8B/configs/meld.json --pattern no_audio
python HREC_ChatGLM3_6B/run.py evaluate --config HREC_ChatGLM3_6B/configs/meld.json --pattern no_text
python HREC_Llama2_7B/run.py evaluate --config HREC_Llama2_7B/configs/meld.json --pattern text_only
python scripts/run_seeds.py --config HREC_Qwen_1_8B/configs/meld.json
```

The repeated-run script retrains both stages for seeds 1111, 2222, 3333, 4444, and 5555. It reports the mean and sample standard deviation for the complete-input test condition. Other patterns can be evaluated using each saved seed configuration.

Classification outputs include accuracy, macro-F1, and weighted-F1. Regression outputs include MAE, Pearson correlation, binary accuracy with and without neutral targets, and seven-class accuracy for MOSEI. Retrieval analysis reports MTD@k and H-nDCG@k. Accuracy and F1 values are fractions in [0,1]. Query labels are accessed for evaluation after retrieval and prediction.

For inference with unlabeled records:

```bash
python HREC_Qwen_1_8B/run.py predict --config HREC_Qwen_1_8B/configs/meld.json --manifest data/qwen/meld/unlabeled.jsonl --output runs/hrec_qwen_meld/predictions.json
```

The prediction manifest omits `label` and `hierarchy`. Outputs contain the prediction, modality weights, retrieval distances, and retrieved record identifiers.

## Run artifacts

| File in `run_dir` | Contents |
|---|---|
| `config.json` | Resolved training configuration |
| `protocol_audit.json` | Manifest hashes, sample counts, source overlap checks |
| `retriever.pt` | Stage 1 parameters, hierarchy, retriever fingerprint |
| `memory.pt`, `memory_build.json` | Frozen coordinates, paired evidence, annotations, provenance checks |
| `task_best.pt` | Selected HREC task parameters and associated configuration |
| `retrieval_history.jsonl`, `task_history.jsonl` | Optimization and validation records |
| `training_report.json` | Selected epoch, model classes, parameter count, freezing checks |
| `test_<pattern>_metrics.json` | Task and retrieval metrics |
| `test_<pattern>_predictions.json` | Per-record predictions and retrieval details |

Memory and task checkpoints are checked against the corresponding retriever and memory hashes. A change to the retriever or feature manifests requires rebuilding dependent artifacts. The generic Slurm template is [scripts/train.slurm](scripts/train.slurm). GPU memory use depends on the backbone, sequence lengths, and micro-batch size.

## Method correspondence and licenses

[docs/METHOD_ALIGNMENT.md](docs/METHOD_ALIGNMENT.md) maps manuscript equations to the implementation and records choices needed where the manuscript does not specify an exact configuration. [VALIDATION.md](VALIDATION.md) reports the scope of the executed checks.

The HREC modules retain the inherited license in [LICENSE](LICENSE). Bundled backbone sources retain their copyright notices and licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Model weights and datasets retain their providers' separate terms.
