"""Shared label tree. Query labels are used for training/evaluation only."""
import math
import torch

# This fixed vocabulary is versioned with every retriever checkpoint.
# Unlisted fine states require an explicit annotation path; no guessed polarity.
SYNONYMS = {
    "happiness": "joy", "happy": "joy", "happier": "joy", "happiest": "joy",
    "sad": "sadness", "angry": "anger", "disgusted": "disgust", "fearful": "fear",
    "surprised": "surprise", "neutrality": "neutral", "positive": "positive",
    "negative": "negative", "惊讶": "surprise", "惊奇": "surprise",
    "高兴": "joy", "开心": "joy", "快乐": "joy", "愤怒": "anger",
    "悲伤": "sadness", "伤心": "sadness", "恐惧": "fear", "害怕": "fear",
    "厌恶": "disgust", "中性": "neutral", "中立": "neutral", "中性情绪": "neutral",
    "平静": "neutral", "calm": "neutral", "happyness": "joy",
}
BASIC_POLARITY = {"joy": "positive", "anger": "negative", "disgust": "negative",
                  "fear": "negative", "sadness": "negative", "neutral": "neutral",
                  "surprise": "undetermined"}
CLASS_NAMES = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]


def canonical(value):
    value = " ".join(str(value).lower().strip().split())
    return SYNONYMS.get(value, value)


def sentiment_level(value, dataset):
    scale = {"mosei": 3.0, "simsv2": 1.0}[dataset]
    value = 3 * float(value) / scale
    # Round ties away from zero; the manuscript does not specify tie handling.
    return max(-3, min(3, int(math.copysign(math.floor(abs(value) + .5), value))))


def annotation_path(record):
    explicit = record.get("hierarchy")
    if explicit:
        path = [canonical(x) for x in explicit]
        if path[0] == "root":
            path = path[1:]
        path = [p for i,p in enumerate(path) if i == 0 or p != path[i-1]]
        if not 1 <= len(path) <= 3 or path[0] not in {"positive", "negative", "neutral", "undetermined"}:
            raise ValueError(f"Invalid hierarchy path for {record['id']}: {path}")
        if len(path) > 1 and path[1] in BASIC_POLARITY and BASIC_POLARITY[path[1]] != path[0]:
            raise ValueError(f"Basic emotion/polarity conflict for {record['id']}")
        return ("root", *path)
    dataset = record.get("dataset", "").lower().replace("-", "")
    label = record.get("label")
    if label is None:
        raise ValueError(f"Missing training/evaluation annotation for {record['id']}")
    if dataset in {"mosei", "simsv2"}:
        level = sentiment_level(label, dataset)
        polarity = "negative" if level < 0 else "positive" if level > 0 else "neutral"
        return ("root", polarity, f"intensity_{abs(level)}")
    if not isinstance(label, str):
        raise ValueError("Categorical labels must be emotion names, not undocumented numeric IDs")
    label = canonical(label)
    if label in BASIC_POLARITY:
        return ("root", "neutral") if label == "neutral" else ("root", BASIC_POLARITY[label], label)
    if label in {"positive", "negative", "undetermined"}:
        return ("root", label)
    raise ValueError(f"Unknown emotion '{label}'; supply a reviewed hierarchy path")


def tree_distance(a, b):
    common = 0
    for left, right in zip(a, b):
        if left != right:
            break
        common += 1
    return len(a) + len(b) - 2 * common


class EmotionTree:
    def __init__(self, records):
        self.paths = [annotation_path(r) for r in records]
        self.nodes = sorted({p[:i] for p in self.paths for i in range(1, len(p)+1)})

    def distances(self, query):
        path = annotation_path(query)
        return torch.tensor([tree_distance(path, p) for p in self.paths], dtype=torch.float32)

    def candidates(self, query, count=96, generator=None):
        if count % 3 or count < 3:
            raise ValueError("candidate count must be a positive multiple of three")
        if len(self.paths) < count:
            raise ValueError(f"Memory needs at least {count} unique candidates, got {len(self.paths)}")
        path = annotation_path(query)
        distances = self.distances(query)
        same = torch.tensor([p[1] == path[1] for p in self.paths])
        nearest = distances == distances.min()
        groups = [nearest, same & ~nearest, ~same & ~nearest]
        selected = []
        for mask in groups:
            pool = torch.where(mask)[0]
            order = torch.randperm(len(pool), generator=generator)
            selected.extend(pool[order[:count//3]].tolist())
        if len(selected) < count:
            remaining = torch.ones(len(distances), dtype=torch.bool)
            remaining[selected] = False
            pool = torch.where(remaining)[0]
            # Shuffle equal-distance ties without replacing any selected entry.
            pool = pool[torch.randperm(len(pool), generator=generator)]
            pool = pool[torch.argsort(distances[pool], stable=True)]
            selected.extend(pool[:count-len(selected)].tolist())
        return torch.tensor(selected), distances[selected]
