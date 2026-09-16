"""JSONL manifests and masked variable-length modality sequences."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

MODALITIES = ("audio", "visual", "text")
PATTERNS = {"all": (1,1,1), "no_text": (1,1,0), "no_visual": (1,0,1),
            "no_audio": (0,1,1), "text_only": (0,0,1)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FeatureDataset(Dataset):
    def __init__(self, manifest, cfg, require_labels=True):
        self.path = Path(manifest).resolve()
        self.cfg = cfg
        self.records = [json.loads(line) for line in self.path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not self.records:
            raise ValueError(f"Empty manifest: {manifest}")
        ids = set()
        for record in self.records:
            for field in ("id", "source_id", "dataset", "split", "feature_space"):
                if not record.get(field):
                    raise ValueError(f"Missing {field} in {manifest}")
            if record["id"] in ids:
                raise ValueError(f"Duplicate sample ID: {record['id']}")
            ids.add(record["id"])
            if require_labels and record.get("label") is None:
                raise ValueError(f"Missing target for {record['id']}")
            if record.get("feature_origin") == "synthetic" and not cfg.get("allow_synthetic", False):
                raise ValueError("Synthetic fixtures require explicit allow_synthetic=true")
        self.manifest_hash = file_sha256(self.path)

    def __len__(self):
        return len(self.records)

    def _array(self, value, modality):
        dim = self.cfg["feature_dims"][modality]
        if value is None:
            return torch.zeros(1, dim)
        path = Path(value)
        if not path.is_absolute():
            path = self.path.parent / path
        array = np.load(path, allow_pickle=False)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[-1] != dim or array.shape[0] < 1:
            raise ValueError(f"Bad {modality} shape at {path}: {array.shape}; expected [time,{dim}]")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite feature at {path}")
        array = array[:self.cfg["max_sequence_length"]]
        return torch.from_numpy(np.array(array, dtype=np.float32, copy=True))

    def __getitem__(self, index):
        record = self.records[index]
        values, observed = {}, []
        for modality in MODALITIES:
            field = "text_features" if modality == "text" else modality
            present = record.get(field) is not None
            if modality == "text" and present and not record.get("text", "").strip():
                raise ValueError("text_features require a non-empty observed text field")
            values[modality] = self._array(record.get(field), modality)
            observed.append(present)
        if not any(observed):
            raise ValueError(f"All modalities missing in {record['id']}")
        return {"features": values, "observed": observed, "record": record}


def collate_features(items):
    features, masks = {}, {}
    for modality in MODALITIES:
        sequences = [item["features"][modality] for item in items]
        lengths = torch.tensor([len(x) for x in sequences])
        features[modality] = pad_sequence(sequences, batch_first=True)
        masks[modality] = torch.arange(features[modality].shape[1])[None] < lengths[:,None]
    return {"features": features, "masks": masks,
            "observed": torch.tensor([x["observed"] for x in items], dtype=torch.bool),
            "records": [x["record"] for x in items]}


def to_device(batch, device):
    return {"features": {k:v.to(device) for k,v in batch["features"].items()},
            "masks": {k:v.to(device) for k,v in batch["masks"].items()},
            "observed": batch["observed"].to(device), "records": batch["records"]}


def apply_pattern(batch, pattern="random", cfg=None):
    batch = {**batch, "features": dict(batch["features"]), "observed": batch["observed"].clone()}
    available = batch["observed"]
    all_patterns = torch.tensor(list(PATTERNS.values()), dtype=torch.bool, device=available.device)
    if pattern == "random":
        # Uniform over five patterns for complete training samples. For naturally
        # incomplete samples, only patterns leaving an observed stream are eligible.
        choices = []
        for row in available:
            eligible = (all_patterns & row).any(-1).nonzero().flatten()
            choice = eligible[torch.randint(len(eligible), (1,), device=row.device)]
            choices.append(all_patterns[choice.item()] & row)
        observed = torch.stack(choices)
    else:
        if pattern not in PATTERNS:
            raise ValueError(f"Unknown modality pattern: {pattern}")
        observed = available & torch.tensor(PATTERNS[pattern], device=available.device, dtype=torch.bool)
    if (~observed.any(-1)).any():
        raise ValueError("Requested pattern leaves a sample with no observed modality")
    batch["observed"] = observed
    for index, modality in enumerate(MODALITIES):
        x = batch["features"][modality]
        if pattern == "random" and cfg:
            std = cfg.get(f"{modality}_noise_std", 0.)
            if std:
                x = x + std * torch.randn_like(x)
        # torch.where also prevents hidden NaNs/features entering the model.
        batch["features"][modality] = torch.where(observed[:,index,None,None], x, torch.zeros_like(x))
    return batch


def validate_protocol(datasets):
    """Reject manifest-level sample/source overlap before any optimization."""
    summaries = {}
    spaces = set()
    for name, dataset in datasets.items():
        expected = "train" if name in {"train", "memory"} else name
        wrong = [r["id"] for r in dataset.records if r["split"] != expected]
        if wrong:
            raise ValueError(f"{name} manifest must contain only split={expected}: {wrong[:3]}")
        spaces.update(r["feature_space"] for r in dataset.records)
        if name == "memory":
            for r in dataset.records:
                if not all(r.get(f) for f in ("audio", "visual", "text_features", "caption")):
                    raise ValueError(f"Memory entry needs paired A/V features, text features and caption: {r['id']}")
                if r.get("feature_origin") not in {"audio_visual_encoders", "synthetic"}:
                    raise ValueError("Memory feature_origin must attest audio_visual_encoders")
                if dataset.cfg.get("require_external_memory", True) and r["dataset"].lower().replace("-", "") != "mercaption":
                    raise ValueError("Paper protocol requires a MER-Caption training memory")
        summaries[name] = {"samples": len(dataset), "manifest_sha256": dataset.manifest_hash}
    if len(spaces) != 1:
        raise ValueError("All datasets must use the same feature_space (extractors, weights, preprocessing)")
    names = list(datasets)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            a, b = datasets[left].records, datasets[right].records
            ids = {r["id"] for r in a} & {r["id"] for r in b}
            sources = {r["source_id"] for r in a} & {r["source_id"] for r in b}
            if ids or sources:
                raise ValueError(f"Overlap between {left} and {right}: ids={len(ids)}, sources={len(sources)}")
    return {"feature_space": list(spaces), "splits": summaries,
            "overlap_check": "passed for supplied IDs/source IDs; not a raw-media duplicate audit"}
