"""Task metrics and retrieval quality measured after label-free ranking."""
import numpy as np
from .hierarchy import annotation_path, tree_distance


def task_metrics(targets, predictions, cfg):
    y = np.asarray(targets)
    p = np.asarray(predictions)
    if cfg["task"] == "classification":
        pred = p.argmax(-1)
        f1, support = [], []
        for label in range(len(cfg["class_names"])):
            tp = np.sum((y == label) & (pred == label))
            fp = np.sum((y != label) & (pred == label))
            fn = np.sum((y == label) & (pred != label))
            f1.append(float(2*tp / max(2*tp+fp+fn, 1)))
            support.append(int(np.sum(y == label)))
        return {"accuracy": float(np.mean(y == pred)), "macro_f1": float(np.mean(f1)),
                "weighted_f1": float(np.average(f1, weights=support)), "class_support": support}
    p = p.reshape(-1)
    corr = float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 and len(y)>1 else None
    result = {"mae": float(np.abs(y-p).mean()), "correlation": corr,
              "binary_accuracy_with_zero": float(np.mean((y>=0) == (p>=0)))}
    nonzero = y != 0
    result["binary_accuracy_nonzero"] = float(np.mean((y[nonzero]>0) == (p[nonzero]>0))) if nonzero.any() else None
    if cfg["dataset"] == "mosei":
        result["accuracy_7"] = float(np.mean(np.round(np.clip(y,-3,3)) == np.round(np.clip(p,-3,3))))
    return result


def retrieval_metrics(queries, selected, memory, tau_h=.5):
    paths = [annotation_path(r) for r in memory]
    k = len(selected[0])
    discount = np.log2(np.arange(2,k+2))
    mtd, ndcg = [], []
    for record, indices in zip(queries, selected):
        query = annotation_path(record)
        all_distances = np.asarray([tree_distance(query,p) for p in paths])
        distances = all_distances[indices]
        ideal = np.sort(all_distances)[:k]
        dcg = np.sum(np.exp(-distances/tau_h)/discount)
        idcg = np.sum(np.exp(-ideal/tau_h)/discount)
        mtd.append(float(distances.mean()))
        ndcg.append(float(dcg / max(idcg, 1e-30)))
    return {f"MTD@{k}": float(np.mean(mtd)), f"H-nDCG@{k}": float(np.mean(ndcg))}
