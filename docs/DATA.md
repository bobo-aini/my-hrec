# Data preparation

## Manifest format

Manifests are UTF-8 JSONL files. Each feature is stored as a non-pickled NumPy `.npy` array of shape `[sequence_length, input_dimension]`; a single one-dimensional vector is also accepted. Modalities may have different sequence lengths. Padding is excluded from encoding and pooling.

Example task record:

```json
{"id":"meld:dia12_utt3","source_id":"meld:episode03_scene12","dataset":"meld","split":"train","text":"I did not expect that.","audio":"features/dia12_utt3_audio.npy","visual":"features/dia12_utt3_visual.npy","text_features":"features/dia12_utt3_text.npy","label":"surprise","feature_space":"audio-v1_visual-v1_text-qwen-v1","feature_origin":"audio_visual_encoders"}
```

Example external memory record:

```json
{"id":"memory:clip0001","source_id":"mercaption:source_video001","dataset":"mercaption","split":"train","text":"What happened here?","caption":"The speaker reacts with surprise to an unexpected event.","audio":"features/clip0001_audio.npy","visual":"features/clip0001_visual.npy","text_features":"features/clip0001_text.npy","label":"surprise","hierarchy":["undetermined","surprise"],"feature_space":"audio-v1_visual-v1_text-qwen-v1","feature_origin":"audio_visual_encoders"}
```

| Field | Definition |
|---|---|
| `id` | Globally unique record identifier |
| `source_id` | Original video or conversation group, assigned before partitioning |
| `dataset` | Corpus identifier: e.g., `meld`, `cherma`, `mosei`, `simsv2`, `mercaption` |
| `split` | `train`, `valid`, or `test`; memory records must use `train` |
| `audio`, `visual` | Feature paths from the stated audio and visual extractors |
| `text` | Observable text without appended target annotations |
| `text_features` | Embeddings of `text` from the selected frozen backbone |
| `label` | Emotion name for classification; numeric regression score; omitted for prediction |
| `caption` | Description of an external memory record |
| `hierarchy` | Optional reviewed path below the root: `[polarity,basic_or_intensity,fine_state]` |
| `feature_space` | Version identifier for the extractors, weights, and preprocessing |
| `feature_origin` | `audio_visual_encoders` for real features; `synthetic` for fixtures |

Feature paths are relative to the manifest, or absolute. A naturally unavailable modality has a `null` path. A query must retain at least one observed modality. Memory records require paired audio, visual, and text features, a caption, and an annotation. Query and memory arrays must originate from the same extractors and preprocessing; matching dimensions alone is insufficient.

## Importing processed benchmarks

Two trusted pickle layouts are supported:

1. Regression: `train/valid/test` dictionaries with `raw_text`, `audio`, `vision`, `regression_labels`, `id`, and optional lengths.
2. Classification: a record list with `features.audio`, `features.video`, optional `features.text`, lengths, and `label`.

Pickle inputs should come from a trusted source because deserialization can execute code. A source map associates each record with its actual video or conversation. For classification files lacking IDs, the importer uses stable lookup keys such as `train_000000`; the mapped values must identify the original sources.

```json
{"train_000000":"meld:episode01_scene02","train_000001":"meld:episode01_scene02"}
```

```bash
python -m hrec convert-benchmark --source /data/MELD/meld_train.pkl --dataset meld --split train --source-map /data/meld_source_map.json --feature-space audio-v1_visual-v1_text-qwen-v1 --output data/qwen/meld/train_raw.jsonl
```

Repeat with validation/test inputs and their respective `--split` and output paths. For numerical classification labels, supply `--label-map /data/label_map.json`, containing the actual numeric-to-emotion mapping. Preserve official splits and source grouping.

## External training memory

Prepare reviewed MER-Caption training annotations with `id`, `source_id`, `split`, `text`, `caption`, `label`, and, when required, `hierarchy`. Supply audio and visual directories containing `<id>.npy` files from the same extractors used for the benchmark.

```bash
python -m hrec import-memory --annotations /data/mercaption_train.jsonl --audio-dir /data/mercaption/audio_features --visual-dir /data/mercaption/visual_features --feature-space audio-v1_visual-v1_text-qwen-v1 --output data/qwen/meld/memory_raw.jsonl
```

The importer verifies paired features and does not generate audio/visual vectors from captions. Unknown fine-grained labels require reviewed paths, e.g., `["negative","anger","frustration"]`. The default objective requires at least 96 distinct eligible memory entries.

## Text features

Set the checkpoint path in the selected configuration and encode all four manifests:

```bash
python HREC_Qwen_1_8B/run.py prepare-text --config HREC_Qwen_1_8B/configs/meld.json --manifest data/qwen/meld/train_raw.jsonl --output data/qwen/meld/train.jsonl
python HREC_Qwen_1_8B/run.py prepare-text --config HREC_Qwen_1_8B/configs/meld.json --manifest data/qwen/meld/valid_raw.jsonl --output data/qwen/meld/valid.jsonl
python HREC_Qwen_1_8B/run.py prepare-text --config HREC_Qwen_1_8B/configs/meld.json --manifest data/qwen/meld/test_raw.jsonl --output data/qwen/meld/test.jsonl
python HREC_Qwen_1_8B/run.py prepare-text --config HREC_Qwen_1_8B/configs/meld.json --manifest data/qwen/meld/memory_raw.jsonl --output data/qwen/meld/memory.jsonl
```

Encoding reads `text` without appending `label` or `hierarchy`. Use the corresponding entry point/configuration for ChatGLM or Llama. Regenerate text features, retriever, and memory when changing the backbone.

## Protocol validation

```bash
python HREC_Qwen_1_8B/run.py audit --config HREC_Qwen_1_8B/configs/meld.json
```

The audit checks manifest structure, dimensions, paired memory evidence, splits, source-ID overlap, and feature-space consistency. These checks operate on supplied metadata; near-duplicate media identification remains a data-preparation step. Exact raw-feature extraction settings, the full feature arrays, external memory annotations, and model weights are required for a complete benchmark reproduction and are not distributed here.
