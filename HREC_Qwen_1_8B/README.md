# HREC with Qwen-1.8B

This variant combines the shared HREC retrieval and completion modules with the Qwen-1.8B architecture. `backbone/` contains the transformer, configuration, and tokenizer source actually loaded by `model.py`. `run.py` validates the backbone selection and provides the training and evaluation commands.

From the repository root, after completing the [environment and data preparation](../README.md):

```bash
python HREC_Qwen_1_8B/run.py pipeline --config HREC_Qwen_1_8B/configs/meld.json
python HREC_Qwen_1_8B/run.py evaluate --config HREC_Qwen_1_8B/configs/meld.json --pattern text_only
```

Available configurations are `meld.json`, `cherma.json`, `mosei.json`, and `simsv2.json`. Set `llm_path`, feature dimensions, manifest paths, and `run_dir` before training. The entry point requires `llm_family="qwen"`. Pretrained weight and tokenizer assets are supplied separately from this source directory.

Implementation details are in [METHOD_ALIGNMENT.md](../docs/METHOD_ALIGNMENT.md), data formats in [DATA.md](../docs/DATA.md), and executed checks in [VALIDATION.md](../VALIDATION.md).
