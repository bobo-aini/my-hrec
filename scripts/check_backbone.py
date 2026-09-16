"""Test a supplied local pretrained backbone with synthetic features on a GPU."""
import argparse
import json
import sys
from pathlib import Path
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hrec.config import load_config
from hrec.demo import create_demo
from hrec.engine import train_retriever,build_memory,train_task,evaluate,save_json

parser = argparse.ArgumentParser()
parser.add_argument("--model",required=True)
parser.add_argument("--family",choices=["qwen","llama","chatglm"],required=True)
parser.add_argument("--directory",default="runs/backbone_check")
args = parser.parse_args()
torch.set_num_threads(4)
cfg = load_config(create_demo(args.directory))
cfg.update({"device":"cuda","llm_backend":"huggingface","llm_family":args.family,"llm_path":args.model,
            "trust_remote_code":False,"llm_dtype":"bfloat16"})
save_json(Path(args.directory)/"config.json",cfg)
train_retriever(cfg)
build_memory(cfg)
training = train_task(cfg)
evaluation = {pattern:evaluate(cfg,pattern=pattern) for pattern in ("all","text_only","no_text")}
save_json(Path(args.directory)/"integration_report.json",{"kind":"real backbone; synthetic features", "training":training,"evaluation":evaluation})
