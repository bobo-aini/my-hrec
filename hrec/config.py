"""Portable configuration; paths in JSON are relative to the project directory."""
import json
from pathlib import Path

DEFAULTS = {
    "seed": 1111, "device": "cuda", "task": "classification", "dataset": "meld",
    "class_names": ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"],
    "feature_dims": {"audio": 768, "visual": 768, "text": 2048},
    "encoder_hidden": 128, "feature_dim": 256, "hyp_dim": 128, "curvature": 1.0,
    "hidden_dim": 128, "attention_dim": 128, "dropout": .1,
    "top_k": 10, "candidates": 96, "tau_h": .5, "tau_r": .5, "tau_a": .1,
    "retrieval_epochs": 5, "task_epochs": 20, "batch_size": 32, "micro_batch_size": 4,
    "learning_rate": 1e-4, "weight_decay": .01, "warmup_ratio": .1, "grad_clip": 1.0,
    "num_workers": 0, "memory_chunk_size": 2048, "max_sequence_length": 512,
    "llm_backend": "huggingface", "llm_path": "pretrained/Qwen-1_8B", "llm_revision": None,
    "llm_family": "qwen", "llm_dtype": "bfloat16", "trust_remote_code": False,
    "local_files_only": True, "max_prompt_tokens": 160, "max_evidence_tokens": 640,
    "tiny_hidden": 48, "tiny_layers": 2, "tiny_heads": 4,
    "train_manifest": "data/train.jsonl", "valid_manifest": "data/valid.jsonl",
    "test_manifest": "data/test.jsonl", "memory_manifest": "data/memory.jsonl",
    "run_dir": "runs/hrec", "audio_noise_std": 0., "visual_noise_std": 0.,
    "text_noise_std": 0., "eval_pattern": "all",
}


def load_config(path):
    cfg = dict(DEFAULTS)
    cfg.update(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    if cfg["task"] not in {"classification", "regression"}:
        raise ValueError("task must be classification or regression")
    for name in ["tau_h", "tau_r", "tau_a", "curvature", "batch_size", "micro_batch_size"]:
        if cfg[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if cfg["top_k"] <= 1:
        raise ValueError("HREC normalized entropy requires top_k > 1")
    if cfg["candidates"] % 3:
        raise ValueError("candidates must be divisible by three")
    if cfg["batch_size"] % cfg["micro_batch_size"]:
        raise ValueError("batch_size must be divisible by micro_batch_size")
    return cfg
