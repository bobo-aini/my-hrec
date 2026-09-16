"""Two-stage optimization, immutable memory, checkpoints and evaluation."""
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from .data import (FeatureDataset, collate_features, to_device, apply_pattern,
                   validate_protocol, file_sha256, MODALITIES)
from .hierarchy import EmotionTree, canonical
from .retrieval import HyperbolicRetriever, module_digest
from .model import build_model
from .metrics import task_metrics, retrieval_metrics


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def safe_load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def run_path(cfg, name):
    directory = Path(cfg["run_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def get_datasets(cfg, names=("train", "valid", "memory")):
    datasets = {name:FeatureDataset(cfg[f"{name}_manifest"],cfg) for name in names}
    audit = validate_protocol(datasets)
    save_json(run_path(cfg,"protocol_audit.json"), audit)
    return datasets


def minibatches(batch, size):
    n = len(batch["records"])
    for start in range(0,n,size):
        end = start+size
        yield {"features": {m:v[start:end] for m,v in batch["features"].items()},
               "masks": {m:v[start:end] for m,v in batch["masks"].items()},
               "observed": batch["observed"][start:end], "records": batch["records"][start:end]}


def optimizer_and_scheduler(parameters, cfg, steps):
    optimizer = torch.optim.AdamW(parameters, lr=cfg["learning_rate"], betas=(.9,.999), weight_decay=cfg["weight_decay"])
    warmup = max(1, math.ceil(steps*cfg["warmup_ratio"]))
    def multiplier(step):
        return (step+1)/warmup if step < warmup else max(0., (steps-step)/max(1,steps-warmup))
    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)


def record_history(cfg, stage, event):
    path = run_path(cfg, f"{stage}_history.jsonl")
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+"\n")
    print(json.dumps({"stage":stage,**event}, ensure_ascii=False), flush=True)


def train_retriever(cfg):
    seed_everything(cfg["seed"])
    if run_path(cfg,"retriever.pt").exists():
        raise FileExistsError("retriever.pt already exists; use a new run_dir for a fresh experiment")
    data = get_datasets(cfg, ("train","valid","test","memory"))
    train, memory = data["train"], data["memory"]
    tree = EmotionTree(memory.records)
    if len(memory) < cfg["candidates"]:
        raise ValueError("Memory smaller than retrieval candidate count")
    model = HyperbolicRetriever(cfg).to(cfg["device"])
    loader = DataLoader(train,batch_size=cfg["batch_size"],shuffle=True,collate_fn=collate_features,num_workers=cfg["num_workers"])
    optimizer, scheduler = optimizer_and_scheduler(model.parameters(),cfg,len(loader)*cfg["retrieval_epochs"])
    save_json(run_path(cfg,"config.json"),cfg)
    initial = module_digest(model)
    for epoch in range(cfg["retrieval_epochs"]):
        model.train()
        total, count = 0.,0
        for step, full in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.
            for batch in minibatches(full,cfg["micro_batch_size"]):
                candidate_items, tree_dist = [],[]
                for record in batch["records"]:
                    indices, distances = tree.candidates(record,cfg["candidates"])
                    candidate_items.extend(memory[int(i)] for i in indices)
                    tree_dist.append(distances)
                candidates = to_device(collate_features(candidate_items),cfg["device"])
                batch = apply_pattern(to_device(batch,cfg["device"]),"random",cfg)
                loss = model.pretraining_loss(batch,candidates,cfg["candidates"],torch.stack(tree_dist).to(cfg["device"]))
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite retrieval loss")
                weight = len(batch["records"]) / len(full["records"])
                (loss*weight).backward()
                loss_sum += loss.item()*weight
            nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += loss_sum*len(full["records"])
            count += len(full["records"])
            if step % 20 == 0:
                record_history(cfg,"retrieval",{"epoch":epoch+1,"step":step+1,"loss":loss_sum})
        record_history(cfg,"retrieval",{"epoch":epoch+1,"epoch_loss":total/count})
    state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    checkpoint = {"format_version":1,"state":state,"config":cfg,"fingerprint":module_digest(model),
                  "initial_fingerprint":initial,"memory_manifest_hash":memory.manifest_hash,
                  "train_manifest_hash":train.manifest_hash,"tree_nodes":[list(p) for p in tree.nodes],
                  "feature_space":sorted({r["feature_space"] for r in memory.records})}
    torch.save(checkpoint,run_path(cfg,"retriever.pt"))
    return checkpoint


RETRIEVER_KEYS = ("feature_dims","feature_dim","encoder_hidden","hyp_dim","curvature","hidden_dim")


def load_retriever(cfg):
    checkpoint = safe_load(run_path(cfg,"retriever.pt"))
    for key in RETRIEVER_KEYS:
        if checkpoint["config"][key] != cfg[key]:
            raise ValueError(f"Retriever configuration mismatch: {key}")
    model = HyperbolicRetriever(cfg)
    model.load_state_dict(checkpoint["state"])
    if module_digest(model) != checkpoint["fingerprint"]:
        raise ValueError("Retriever checkpoint fingerprint mismatch")
    model.requires_grad_(False).eval().to(cfg["device"])
    return model,checkpoint


@torch.no_grad()
def build_memory(cfg):
    model, checkpoint = load_retriever(cfg)
    data = get_datasets(cfg,("train","valid","test","memory"))
    memory = data["memory"]
    if checkpoint["memory_manifest_hash"] != memory.manifest_hash:
        raise ValueError("Memory manifest changed after retriever training")
    coordinates, audio, visual = [],[],[]
    loader = DataLoader(memory,batch_size=cfg["batch_size"],shuffle=False,collate_fn=collate_features)
    for batch in loader:
        batch = to_device(batch,cfg["device"])
        _, summaries, tangent = model.encode(batch)
        coordinates.append(model.ball.exp_map_zero(tangent).cpu())
        audio.append(summaries["audio"].cpu())
        visual.append(summaries["visual"].cpu())
    records = [{k:r[k] for k in ("id","source_id","dataset","split","label","caption","feature_space","hierarchy") if k in r} for r in memory.records]
    bank = {"format_version":1,"coordinates":torch.cat(coordinates),
            "evidence":{"audio":torch.cat(audio),"visual":torch.cat(visual)},"records":records,
            "retriever_fingerprint":checkpoint["fingerprint"],"memory_manifest_hash":memory.manifest_hash}
    torch.save(bank,run_path(cfg,"memory.pt"))
    report = {"entries":len(memory),"retriever_fingerprint":checkpoint["fingerprint"],
              "memory_sha256":file_sha256(run_path(cfg,"memory.pt")),"coordinates_shape":list(bank["coordinates"].shape)}
    save_json(run_path(cfg,"memory_build.json"),report)
    print(json.dumps(report),flush=True)
    return bank


def load_bank(cfg, checkpoint):
    bank = safe_load(run_path(cfg,"memory.pt"))
    if bank["retriever_fingerprint"] != checkpoint["fingerprint"]:
        raise ValueError("Stale memory: retriever fingerprint differs; rebuild memory")
    return bank


def targets(batch,cfg,device):
    if cfg["task"] == "classification":
        mapping = {canonical(name):i for i,name in enumerate(cfg["class_names"])}
        values = [mapping[canonical(r["label"])] for r in batch["records"]]
        return torch.tensor(values,dtype=torch.long,device=device)
    return torch.tensor([float(r["label"]) for r in batch["records"]],device=device)


def task_loss(prediction,target,cfg):
    return nn.functional.cross_entropy(prediction,target) if cfg["task"] == "classification" else nn.functional.smooth_l1_loss(prediction,target)


@torch.no_grad()
def evaluate_model(model,dataset,cfg,pattern="all",quality=False):
    model.eval()
    losses, actual, predictions, records, selected, details = [],[],[],[],[],[]
    loader = DataLoader(dataset,batch_size=cfg["micro_batch_size"],shuffle=False,collate_fn=collate_features)
    for batch in loader:
        batch = apply_pattern(to_device(batch,cfg["device"]),pattern)
        result = model(batch)
        target = targets(batch,cfg,cfg["device"])
        loss = task_loss(result["prediction"],target,cfg)
        losses.extend([loss.item()]*len(batch["records"]))
        actual.extend(target.cpu().tolist())
        predictions.extend(result["prediction"].cpu().tolist())
        records.extend(batch["records"])
        selected.extend(result["retrieval"]["indices"].tolist())
        for b,record in enumerate(batch["records"]):
            details.append({"id":record["id"],"prediction":result["prediction"][b].cpu().tolist(),
                            "beta":result["beta"][b].cpu().tolist(),"alpha":result["retrieval"]["alpha"][b].cpu().tolist(),
                            "distances":result["retrieval"]["distances"][b].cpu().tolist(),
                            "retrieved_ids":[model.bank["records"][j]["id"] for j in selected[-len(batch["records"])+b]]})
    report = {"samples":len(dataset),"loss":float(np.mean(losses)),"pattern":pattern,
              **task_metrics(actual,predictions,cfg)}
    if quality:
        report.update(retrieval_metrics(records,selected,model.bank["records"],cfg["tau_h"]))
    return report,details


def task_state(model):
    return {k:v.detach().cpu() for k,v in model.state_dict().items()
            if not k.startswith("retriever.") and (model.cfg["llm_backend"] == "tiny" or not k.startswith("backbone.model."))}


def train_task(cfg):
    seed_everything(cfg["seed"])
    if run_path(cfg,"task_best.pt").exists():
        raise FileExistsError("task_best.pt exists; use a new run_dir to avoid overwriting an experiment")
    data = get_datasets(cfg,("train","valid","test","memory"))
    retriever, checkpoint = load_retriever(cfg)
    if data["train"].manifest_hash != checkpoint["train_manifest_hash"]:
        raise ValueError("Task training manifest differs from retrieval training manifest")
    bank = load_bank(cfg,checkpoint)
    if bank["memory_manifest_hash"] != data["memory"].manifest_hash:
        raise ValueError("Memory manifest changed; retrain and rebuild")
    model = build_model(cfg,retriever,bank).to(cfg["device"])
    parameters = [p for p in model.parameters() if p.requires_grad]
    loader = DataLoader(data["train"],batch_size=cfg["batch_size"],shuffle=True,collate_fn=collate_features)
    optimizer, scheduler = optimizer_and_scheduler(parameters,cfg,len(loader)*cfg["task_epochs"])
    bank_hash = file_sha256(run_path(cfg,"memory.pt"))
    initial_completion = module_digest(model.completion)
    best, best_epoch = float("inf"), 0
    for epoch in range(cfg["task_epochs"]):
        model.train()
        total,count = 0.,0
        for step,full in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.
            for batch in minibatches(full,cfg["micro_batch_size"]):
                batch = apply_pattern(to_device(batch,cfg["device"]),"random",cfg)
                prediction = model(batch)["prediction"]
                loss = task_loss(prediction,targets(batch,cfg,cfg["device"]),cfg)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite task loss")
                weight = len(batch["records"])/len(full["records"])
                (loss*weight).backward()
                total_loss += loss.item()*weight
            if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.completion.parameters()):
                raise RuntimeError("No completion gradient through the frozen language model")
            nn.utils.clip_grad_norm_(parameters,cfg["grad_clip"],error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            total += total_loss*len(full["records"])
            count += len(full["records"])
            if step%20 == 0:
                record_history(cfg,"task",{"epoch":epoch+1,"step":step+1,"loss":total_loss})
        validation,_ = evaluate_model(model,data["valid"],cfg)
        event = {"epoch":epoch+1,"train_loss":total/count,"validation":validation}
        record_history(cfg,"task",event)
        if validation["loss"] < best:
            best,best_epoch = validation["loss"],epoch+1
            torch.save({"format_version":1,"state":task_state(model),"config":cfg,
                        "retriever_fingerprint":checkpoint["fingerprint"],"memory_sha256":bank_hash,
                        "best_epoch":best_epoch,"validation":validation},run_path(cfg,"task_best.pt"))
    if module_digest(model.retriever) != checkpoint["fingerprint"]:
        raise RuntimeError("Frozen retriever changed during task training")
    report = {"model_class":type(model).__module__+"."+type(model).__name__,
              "backbone_class":type(model.backbone.model).__module__+"."+type(model.backbone.model).__name__,
              "best_epoch":best_epoch,"best_validation_loss":best,
              "retriever_unchanged":True,"completion_changed":module_digest(model.completion)!=initial_completion,
              "trainable_parameters":sum(p.numel() for p in parameters),"memory_sha256":bank_hash}
    save_json(run_path(cfg,"training_report.json"),report)
    return report


def restore_task(cfg):
    checkpoint = safe_load(run_path(cfg,"task_best.pt"))
    for key in (*RETRIEVER_KEYS,"task","class_names","attention_dim","llm_backend","llm_family","llm_path","llm_revision","tiny_hidden","tiny_layers","tiny_heads","top_k","tau_a"):
        if checkpoint["config"].get(key) != cfg.get(key):
            raise ValueError(f"Task checkpoint configuration mismatch: {key}")
    retriever, retrieval_checkpoint = load_retriever(cfg)
    if checkpoint["retriever_fingerprint"] != retrieval_checkpoint["fingerprint"]:
        raise ValueError("Task/retriever mismatch")
    if file_sha256(run_path(cfg,"memory.pt")) != checkpoint["memory_sha256"]:
        raise ValueError("Task/memory mismatch")
    model = build_model(cfg,retriever,load_bank(cfg,retrieval_checkpoint)).to(cfg["device"])
    result = model.load_state_dict(checkpoint["state"],strict=False)
    unexpected = result.unexpected_keys
    missing = [k for k in result.missing_keys if not (k.startswith("retriever.") or k.startswith("backbone.model."))]
    if unexpected or missing:
        raise ValueError(f"Invalid task checkpoint: missing={missing}, unexpected={unexpected}")
    return model.eval()


def evaluate(cfg,split="test",pattern="all"):
    data = get_datasets(cfg,("train","valid","test","memory"))
    model = restore_task(cfg)
    report,details = evaluate_model(model,data[split],cfg,pattern,quality=True)
    save_json(run_path(cfg,f"{split}_{pattern}_metrics.json"),report)
    save_json(run_path(cfg,f"{split}_{pattern}_predictions.json"),details)
    print(json.dumps(report),flush=True)
    return report
