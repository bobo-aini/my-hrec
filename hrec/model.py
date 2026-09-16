"""Geodesic-guided completion and geodesic-conditioned modality balancing."""
import math
import torch
from torch import nn
from .data import MODALITIES
from .retrieval import masked_pool, mlp
from .backbone import FrozenLanguageModel


class EvidenceCompletion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim, hidden = cfg["feature_dim"], cfg["attention_dim"]
        self.query = nn.Linear(dim, hidden, bias=False)
        self.key = nn.Linear(dim, hidden, bias=False)
        self.value = nn.Linear(dim, hidden, bias=False)
        self.output = nn.Linear(hidden, dim, bias=False)
        self.missing = nn.Parameter(torch.zeros(1,1,dim))
        self.gate = mlp(2*dim+1, cfg["hidden_dim"], 1)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.scale = math.sqrt(hidden)

    def forward(self, sequence, mask, observed, evidence, log_weights):
        present = observed[:,None,None]
        current = torch.where(present, sequence, self.missing.expand_as(sequence))
        # A missing sequence is a single learned token, not repeated valid padding.
        missing_mask = torch.zeros_like(mask)
        missing_mask[:,0] = True
        valid = torch.where(observed[:,None], mask, missing_mask)
        content = self.query(current) @ self.key(evidence).transpose(1,2) / self.scale
        attention = (content + log_weights[:,None]).softmax(-1)
        update = self.output(self.dropout(attention) @ self.value(evidence))
        gate_input = torch.cat([masked_pool(current, valid), masked_pool(update, valid), observed.float()[:,None]], -1)
        eta = torch.sigmoid(self.gate(gate_input))
        completed = current + eta[:,:,None]*update
        return completed, valid, eta, attention


class HREC(nn.Module):
    def __init__(self, cfg, retriever, bank, backbone=None):
        super().__init__()
        self.cfg, self.retriever, self.bank = cfg, retriever, bank
        self.retriever.requires_grad_(False)
        self.retriever.eval()
        self.backbone = backbone if backbone is not None else FrozenLanguageModel(cfg)
        dim = cfg["feature_dim"]
        self.completion = nn.ModuleDict({m:EvidenceCompletion(cfg) for m in ("audio", "visual")})
        self.missing_text = nn.Parameter(torch.zeros(1,dim))
        self.balancing = mlp(3*dim+5, cfg["hidden_dim"], 3)
        self.fusion = nn.ModuleDict({m:nn.Linear(dim,dim,bias=False) for m in MODALITIES})
        self.alignment = nn.Linear(dim, self.backbone.hidden_dim, bias=False)
        outputs = len(cfg["class_names"]) if cfg["task"] == "classification" else 1
        self.head = nn.Linear(self.backbone.hidden_dim, outputs)

    def train(self, mode=True):
        super().train(mode)
        self.retriever.eval()
        self.backbone.eval()
        return self

    def forward(self, batch):
        retrieved = self.retriever.retrieve(batch, self.bank)
        distances = retrieved["distances"]
        log_weights = (-distances/self.cfg["tau_a"]).log_softmax(-1)
        weights = log_weights.exp()
        summaries, eta, attention = {}, {}, {}
        for i, modality in enumerate(("audio", "visual")):
            sequence, mask, eta[modality], attention[modality] = self.completion[modality](
                retrieved["sequences"][modality], batch["masks"][modality], batch["observed"][:,i],
                retrieved["evidence"][modality], log_weights)
            summaries[modality] = masked_pool(sequence, mask)
        summaries["text"] = torch.where(batch["observed"][:,2,None], retrieved["summaries"]["text"], self.missing_text)
        mean_distance = (weights*distances).sum(-1,keepdim=True)
        entropy = -(weights*log_weights).sum(-1,keepdim=True) / math.log(distances.shape[1])
        descriptor = torch.cat([mean_distance, entropy, batch["observed"].float()], -1)
        beta = self.balancing(torch.cat([*[summaries[m] for m in MODALITIES], descriptor], -1)).softmax(-1)
        fused = sum(beta[:,i,None]*self.fusion[m](summaries[m]) for i,m in enumerate(MODALITIES))
        prefix = self.alignment(fused)
        prompts, evidence = self.backbone.format_inputs(batch, retrieved, self.bank)
        hidden = self.backbone(prefix, prompts, evidence)
        prediction = self.head(hidden)
        if self.cfg["task"] == "regression":
            prediction = prediction.squeeze(-1)
        return {"prediction": prediction, "beta": beta, "eta": eta, "descriptor": descriptor,
                "retrieval": retrieved, "attention": attention, "prefix": prefix}


def build_model(cfg, retriever, bank):
    """Construct the selected, explicitly named HREC model variant."""
    if cfg["llm_backend"] == "tiny":
        return HREC(cfg, retriever, bank)
    from importlib import import_module
    variants = {"qwen":("HREC_Qwen_1_8B", "HRECQwen"),
                "chatglm":("HREC_ChatGLM3_6B", "HRECChatGLM"),
                "llama":("HREC_Llama2_7B", "HRECLlama")}
    package, name = variants[cfg["llm_family"]]
    return getattr(import_module(package+".model"), name)(cfg, retriever, bank)
