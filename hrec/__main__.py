"""python -m hrec --help"""
import argparse
import json
from pathlib import Path
import torch
from .config import load_config
from .engine import (get_datasets,train_retriever,build_memory,train_task,evaluate,
                     restore_task,save_json,run_path)
from .data import FeatureDataset,collate_features,to_device,apply_pattern,PATTERNS


def main(argv=None, expected_family=None):
    parser = argparse.ArgumentParser(description="HREC training, evaluation and feature preparation")
    sub = parser.add_subparsers(dest="command",required=True)
    for name in ("pipeline","audit","pretrain","build-memory","train","evaluate","predict","prepare-text"):
        command = sub.add_parser(name)
        command.add_argument("--config",required=True)
        if name == "evaluate":
            command.add_argument("--split",choices=["valid","test"],default="test")
            command.add_argument("--pattern",choices=list(PATTERNS),default="all")
        if name in {"predict","prepare-text"}:
            command.add_argument("--manifest",required=True)
            command.add_argument("--output",required=True)
    demo = sub.add_parser("demo",help="Complete two-stage synthetic smoke run with a tiny frozen model")
    demo.add_argument("--directory",default="runs/demo")
    demo.add_argument("--task",choices=["classification","regression"],default="classification")
    convert = sub.add_parser("convert-benchmark")
    for name in ("source","dataset","split","output","feature-space"):
        convert.add_argument("--"+name,required=True)
    convert.add_argument("--source-map")
    convert.add_argument("--label-map")
    convert.add_argument("--limit",type=int)
    memory = sub.add_parser("import-memory")
    for name in ("annotations","audio-dir","visual-dir","output","feature-space"):
        memory.add_argument("--"+name,required=True)
    args = parser.parse_args(argv)
    if args.command == "demo":
        from .demo import create_demo
        torch.set_num_threads(2)
        cfg = load_config(create_demo(args.directory,args.task))
        train_retriever(cfg)
        build_memory(cfg)
        train_task(cfg)
        reports = {pattern:evaluate(cfg,pattern=pattern) for pattern in PATTERNS}
        save_json(run_path(cfg,"smoke_report.json"),{"kind":"synthetic pipeline validation","results":reports})
        return
    if args.command == "convert-benchmark":
        from .prepare import convert_benchmark
        convert_benchmark(args.source,args.dataset,args.split,args.output,args.feature_space,args.source_map,args.label_map,args.limit)
        return
    if args.command == "import-memory":
        from .prepare import import_memory
        import_memory(args.annotations,args.audio_dir,args.visual_dir,args.output,args.feature_space)
        return
    cfg = load_config(args.config)
    if expected_family is not None and cfg["llm_family"] != expected_family:
        parser.error(f"This entry point requires llm_family={expected_family}")
    if args.command == "pipeline":
        import subprocess
        import sys
        for stage in ("audit", "pretrain", "build-memory", "train", "evaluate"):
            subprocess.run([sys.executable,"-m","hrec",stage,"--config",args.config],check=True)
        return
    if args.command == "prepare-text":
        from .prepare import prepare_text
        prepare_text(args.manifest,args.output,cfg)
    elif args.command == "audit":
        datasets = get_datasets(cfg,("train","valid","test","memory"))
        for dataset in datasets.values():
            for index in range(len(dataset)):
                dataset[index]
        print("Manifest, overlap and feature validation passed",flush=True)
    elif args.command == "pretrain":
        train_retriever(cfg)
    elif args.command == "build-memory":
        build_memory(cfg)
    elif args.command == "train":
        train_task(cfg)
    elif args.command == "evaluate":
        evaluate(cfg,args.split,args.pattern)
    elif args.command == "predict":
        dataset = FeatureDataset(args.manifest,cfg,require_labels=False)
        model = restore_task(cfg)
        expected_spaces = {r["feature_space"] for r in model.bank["records"]}
        if {r["feature_space"] for r in dataset.records} != expected_spaces:
            raise ValueError("Prediction feature_space differs from the frozen memory")
        results = []
        with torch.no_grad():
            for start in range(0,len(dataset),cfg["micro_batch_size"]):
                batch = collate_features([dataset[i] for i in range(start,min(start+cfg["micro_batch_size"],len(dataset)))])
                batch = apply_pattern(to_device(batch,cfg["device"]),"all")
                output = model(batch)
                for b,row in enumerate(batch["records"]):
                    pred = output["prediction"][b].cpu()
                    results.append({"id":row["id"],"prediction":cfg["class_names"][pred.argmax().item()] if cfg["task"]=="classification" else pred.item(),
                                    "beta":output["beta"][b].cpu().tolist(),"distances":output["retrieval"]["distances"][b].cpu().tolist(),
                                    "retrieved_ids":[model.bank["records"][i]["id"] for i in output["retrieval"]["indices"][b].tolist()]})
        save_json(args.output,results)


if __name__ == "__main__":
    main()
