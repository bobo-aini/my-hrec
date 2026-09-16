# Software validation

Validation date: 2026-09-09. The checks below were executed on Slurm compute nodes. They assess software execution, input interfaces, and gradient behavior; synthetic-data metrics are not benchmark results.

## Current source release

| Check | Result and scope | Record |
|---|---|---|
| Mechanism and backbone tests | 23 tests passed: geometry, hierarchy, masking, retrieval, completion, label independence, frozen parameters, padding, data checks, and the three bundled transformer sources | `validation/revision_unit_tests.txt` |
| Qwen-1.8B | Full pretrained weights loaded through the bundled Qwen implementation; synthetic-feature retrieval pretraining, memory construction, task training, checkpoint reload, and three observation patterns completed | `validation/revision_qwen_integration.json` |
| ChatGLM3 | Reduced random architecture; local tokenizer/checkpoint loading, separate entry point, both training stages, checkpoint reload, five observation patterns, and target-label independence passed | `validation/revision_local_backbones.json` |
| Llama2 | Reduced random architecture; local tokenizer/checkpoint loading, separate entry point, both training stages, checkpoint reload, five observation patterns, and target-label independence passed | `validation/revision_local_backbones.json` |

The reduced ChatGLM and Llama checks use two transformer layers, hidden dimension 32, and synthetic features. They exercise the distributed implementations and native tokenizers, not the full 6B/7B pretrained models. The Qwen check uses the full pretrained 1.8B model with small synthetic features and short sequences.

The source tests verify nonzero gradients to the multimodal prefix, absence of gradients to frozen parameters, and consistency between padded batches and individual examples. The integration reports record the actual HREC and transformer class names. Package integrity and clean-extraction results are provided in the separate release verification report next to the ZIP archive.

## Earlier shared-method checks

These records validate the shared implementation before the model sources were incorporated into the three explicit directories:

| Check | Scope | Record |
|---|---|---|
| Classification and regression | Complete two-stage synthetic runs and five observation patterns | `validation/synthetic_classification.json`, `validation/synthetic_regression.json` |
| Unlabeled CLI and data import | Classification/regression prediction without query labels; two processed MELD records imported for I/O validation | `validation/release_validation.json` |
| Paper-sized core modules | Pretrained Qwen with 128-dimensional geometry, 256-dimensional features, 128-dimensional attention, 96 candidates, top-10 retrieval; one update in each stage | `validation/paper_shape_check.json` |
| Qwen text preparation | Finite 2048-dimensional embeddings from the actual pretrained model | `validation/release_validation.json` |

Earlier reduced backbone interface tests are retained in `validation/backbone_contracts.json`; the current release additionally tests the bundled sources and local loading as listed above.

## Environment and interpretation

Python 3.10.16; PyTorch 2.6.0 with CUDA 12.4; NumPy 1.26.4; Transformers 4.40.2; Tokenizers 0.19.1; Accelerate 0.34.2. GPU checks used an NVIDIA H100 80 GB. Additional package versions are listed in `requirements-tested.txt`. Machine-specific path prefixes in distributed logs are replaced with neutral placeholders; test outcomes are unchanged.

Complete benchmark training across the four datasets and five seeds has not been performed for this release. The manuscript's reported experimental values have not been reproduced or independently confirmed. Full pretrained ChatGLM3-6B and Llama2-7B training has not been executed. The archive does not supply a complete real external feature memory, raw audio/video extraction assets, or benchmark-trained HREC checkpoints.

The executable protocol requires prepared benchmark features, reviewed external training annotations, consistent source grouping, and the selected pretrained weights. Software checks establish functionality within their reported scope and do not replace these experimental prerequisites.
