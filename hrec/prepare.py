"""Explicit feature import. Audio/visual evidence is never synthesized from text."""
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from .backbone import FrozenLanguageModel
from .data import file_sha256
from .hierarchy import canonical, annotation_path


def write_jsonl(path, records):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(r,ensure_ascii=False) for r in records)+"\n",encoding="utf-8")


def clean_array(value,length=None):
    value = np.asarray(value,dtype=np.float32)
    if value.ndim == 1:
        value = value[None]
    if length is not None:
        value = value[:int(length)]
    if value.ndim != 2 or not len(value) or not np.isfinite(value).all():
        raise ValueError("Features must be finite, nonempty [time, channels] arrays")
    return value


def convert_benchmark(source,dataset,split,output,feature_space,source_map=None,label_map=None,limit=None):
    """Read a trusted processed benchmark pickle, keeping task labels separate."""
    source, output = Path(source),Path(output)
    if output.exists():
        raise FileExistsError(output)
    with source.open("rb") as stream:
        raw = pickle.load(stream)
    sources = json.loads(Path(source_map).read_text(encoding="utf-8-sig")) if source_map else {}
    labels = json.loads(Path(label_map).read_text(encoding="utf-8-sig")) if label_map else {}
    out_dir = output.parent / (output.stem+"_features")
    out_dir.mkdir(parents=True,exist_ok=True)
    rows = []
    if isinstance(raw,dict) and split in raw:
        data = raw[split]
        for i,text in enumerate(data["raw_text"]):
            rid = data.get("id",list(range(len(data["raw_text"]))))[i]
            rid = json.dumps(np.asarray(rid).tolist(),ensure_ascii=False) if isinstance(rid,(list,tuple,np.ndarray)) else str(rid)
            rows.append({"id":rid,"text":str(text),"label":float(np.asarray(data["regression_labels"][i]).reshape(-1)[0]),
                         "audio":data["audio"][i],"visual":data["vision"][i],
                         "audio_len":data.get("audio_lengths",[None]*len(data["raw_text"]))[i],
                         "visual_len":data.get("vision_lengths",[None]*len(data["raw_text"]))[i]})
    elif isinstance(raw,list):
        for i,item in enumerate(raw):
            features = item["features"]
            rid = str(item.get("id",item.get("name",f"{split}_{i:06d}")))
            label = item["label"]
            if not isinstance(label,str) or str(label).isdigit():
                if str(label) not in labels:
                    raise ValueError("Numeric categorical labels require --label-map JSON")
                label = labels[str(label)]
            rows.append({"id":rid,"source_id":item.get("source_id"),"text":str(features["text"]),"label":canonical(label),
                         "audio":features["audio"],"visual":features["video"],
                         "audio_len":features.get("audio_len"),"visual_len":features.get("video_len")})
    else:
        raise ValueError("Expected a split dictionary or list of feature records")
    records = []
    for i,row in enumerate(rows[:limit]):
        source_id = sources.get(row["id"],row.get("source_id"))
        if not source_id:
            raise ValueError(f"Source group unavailable for ID {row['id']}; supply --source-map (original ID -> corpus:video/dialogue)")
        record = {"id":f"{dataset}:{row['id']}","source_id":source_id,"dataset":dataset,"split":split,
                  "text":row["text"],"label":row["label"],"feature_space":feature_space,
                  "feature_origin":"audio_visual_encoders"}
        for modality in ("audio","visual"):
            path = out_dir/f"{i:07d}_{modality}.npy"
            np.save(path,clean_array(row[modality],row.get(f"{modality}_len")))
            record[modality] = str(path.resolve())
        annotation_path(record)
        records.append(record)
    write_jsonl(output,records)
    print(json.dumps({"records":len(records),"output":str(output),"text_features":"run prepare-text next"}),flush=True)


def prepare_text(manifest,output,cfg):
    if cfg["llm_backend"] != "huggingface":
        raise ValueError("Production text preparation requires a real pretrained language model")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    manifest = Path(manifest).resolve()
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    model = FrozenLanguageModel(cfg).to(cfg["device"]).eval()
    if model.hidden_dim != cfg["feature_dims"]["text"]:
        raise ValueError("feature_dims.text does not match the selected backbone embedding dimension")
    directory = output.parent/(output.stem+"_text")
    directory.mkdir(parents=True,exist_ok=True)
    with torch.no_grad():
        for i,record in enumerate(records):
            # Only the observed text field is read. Caption/label/hierarchy are not concatenated.
            text = record.get("text","")
            if not text.strip():
                record["text_features"] = None
            else:
                path = directory/f"{i:07d}.npy"
                np.save(path,model.text_features(text,cfg["max_sequence_length"]).numpy())
                record["text_features"] = str(path.resolve())
            for modality in ("audio","visual"):
                if record.get(modality):
                    path = Path(record[modality])
                    record[modality] = str(path if path.is_absolute() else (manifest.parent/path).resolve())
            if i%500==0:
                print(f"Encoded text {i}/{len(records)}",flush=True)
    write_jsonl(output,records)
    metadata = {"input_manifest_sha256":file_sha256(manifest),"backbone":cfg["llm_path"],
                "revision":cfg.get("llm_revision"),"embedding_dim":model.hidden_dim,
                "feature_space":sorted({r["feature_space"] for r in records}),
                "input_field":"text only; no appended annotation fields"}
    output.with_suffix(".meta.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")


def import_memory(annotations,audio_dir,visual_dir,output,feature_space):
    """Join reviewed MER-Caption train annotations with actual extracted features."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    rows = [json.loads(line) for line in Path(annotations).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    result = []
    for row in rows:
        for key in ("id","source_id","split","text","caption","label","hierarchy"):
            if key not in row:
                raise ValueError(f"Memory annotation missing {key}")
        if row["split"] != "train":
            raise ValueError("Only training entries can enter the external memory")
        annotation_path(row)
        record = {**row,"dataset":"mercaption","feature_space":feature_space,"feature_origin":"audio_visual_encoders"}
        for modality,directory in (("audio",audio_dir),("visual",visual_dir)):
            path = Path(directory)/(row["id"]+".npy")
            if not path.is_file():
                raise FileNotFoundError(f"Missing paired {modality} feature: {path}")
            clean_array(np.load(path,allow_pickle=False))
            record[modality] = str(path.resolve())
        result.append(record)
    write_jsonl(output,result)
    print(json.dumps({"records":len(result),"output":str(output),"text_features":"run prepare-text next"}),flush=True)
