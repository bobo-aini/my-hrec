"""Small explicitly synthetic fixtures for installation and gradient checks."""
import json
from pathlib import Path
import numpy as np
from .config import DEFAULTS
from .engine import save_json


def create_demo(directory, task="classification"):
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    arrays = directory / "features"
    arrays.mkdir(exist_ok=True)
    rng = np.random.default_rng(20260909)
    classes = ["neutral","joy","sadness"]
    for split,size in (("train",12),("valid",6),("test",6),("memory",24)):
        records = []
        for i in range(size):
            kind = i%3
            identity = f"{split}_{i:04d}"
            record = {"id":identity,"source_id":f"fixture:{identity}","dataset":"mercaption" if split=="memory" else ("mosei" if task=="regression" else "meld"),
                      "split":"train" if split=="memory" else split,"text":"A speaker describes an event.",
                      "caption":"A speaker reacts to an event.","label":classes[kind],
                      "feature_space":"hrec_synthetic_v1","feature_origin":"synthetic"}
            if task == "regression":
                record["label"] = [0.,2.,-2.][kind]
                if split == "memory":
                    polarity = ["neutral","positive","negative"][kind]
                    record["hierarchy"] = [polarity,f"intensity_{abs(int(record['label']))}"]
            for modality,dim in (("audio",8),("visual",10),("text",12)):
                name = f"{identity}_{modality}.npy"
                feature = rng.normal(0,.2,(3+i%4,dim)).astype(np.float32)
                feature[:,kind] += .4
                np.save(arrays/name,feature)
                record["text_features" if modality=="text" else modality] = f"features/{name}"
            records.append(record)
        (directory/f"{split}.jsonl").write_text("\n".join(json.dumps(x) for x in records)+"\n",encoding="utf-8")
    cfg = {**DEFAULTS,"task":task,"dataset":"mosei" if task=="regression" else "meld",
           "device":"cpu","llm_backend":"tiny","class_names":classes,
           "feature_dims":{"audio":8,"visual":10,"text":12},"feature_dim":16,
           "encoder_hidden":12,"hyp_dim":8,"hidden_dim":16,"attention_dim":8,
           "top_k":4,"candidates":12,"retrieval_epochs":1,"task_epochs":1,
           "batch_size":4,"micro_batch_size":2,"tiny_hidden":24,"tiny_heads":4,"tiny_layers":1,
           "max_prompt_tokens":64,"max_evidence_tokens":384,"max_sequence_length":16,
           "memory_chunk_size":7,"allow_synthetic":True,"run_dir":str(directory/"run")}
    for name in ("train","valid","test","memory"):
        cfg[f"{name}_manifest"] = str(directory/f"{name}.jsonl")
    save_json(directory/"config.json",cfg)
    return directory/"config.json"
