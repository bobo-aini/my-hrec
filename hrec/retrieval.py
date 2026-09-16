"""Hierarchy-supervised retrieval with identical query/memory modality weights."""
import hashlib
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .data import MODALITIES
from .geometry import PoincareBall


def masked_pool(x, mask):
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(1) / mask.sum(1).clamp_min(1)


def mlp(input_dim, hidden_dim, output_dim):
    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))


def module_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class SequenceEncoder(nn.Module):
    """Length-aware recurrent encoder retaining a channel-projected sequence."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, mask):
        packed = pack_padded_sequence(x, mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        encoded, _ = self.rnn(packed)
        encoded, _ = pad_packed_sequence(encoded, batch_first=True, total_length=x.shape[1])
        return self.projection(encoded) * mask.unsqueeze(-1)


class HyperbolicRetriever(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["feature_dim"]
        self.ball = PoincareBall(cfg["curvature"])
        self.encoders = nn.ModuleDict({m: SequenceEncoder(cfg["feature_dims"][m], cfg["encoder_hidden"], d) for m in MODALITIES})
        self.projectors = nn.ModuleDict({m: nn.Linear(d, cfg["hyp_dim"]) for m in MODALITIES})
        self.weight_network = mlp(3*d+3, cfg["hidden_dim"], 3)
        # Small initialization avoids starting at the saturated ball boundary.
        for head in self.projectors.values():
            nn.init.normal_(head.weight, std=.01)
            nn.init.zeros_(head.bias)

    def encode(self, batch):
        sequences = {m:self.encoders[m](batch["features"][m], batch["masks"][m]) for m in MODALITIES}
        summaries = {m:masked_pool(sequences[m], batch["masks"][m]) for m in MODALITIES}
        tangent = torch.stack([self.projectors[m](summaries[m]) for m in MODALITIES], 1)
        return sequences, summaries, tangent

    def query(self, summaries, tangent, observed):
        eligible = observed.clone()
        eligible[:,2] = ~observed[:,0] & ~observed[:,1] & observed[:,2]
        if (~eligible.any(1)).any():
            raise ValueError("Empty retrieval query")
        stack = torch.stack([summaries[m] for m in MODALITIES], 1)
        clean = torch.where(eligible[:,:,None], stack, torch.zeros_like(stack))
        logits = self.weight_network(torch.cat([clean.flatten(1), eligible.float()], -1))
        alpha = logits.masked_fill(~eligible, -torch.inf).softmax(-1)
        z = torch.where(eligible[:,:,None], tangent, torch.zeros_like(tangent))
        # Apply the exp/log pair on both sides to share the numerical domain.
        z = self.ball.log_map_zero(self.ball.exp_map_zero(z))
        query = self.ball.exp_map_zero((alpha[:,:,None] * z).sum(1))
        return query, alpha

    def candidate_distances(self, query, alpha, candidate_tangent):
        # candidate_tangent: [B,C,3,d] for online candidates or [C,3,d] for a bank.
        if candidate_tangent.ndim == 3:
            candidate_tangent = candidate_tangent[None]
        tangent = self.ball.log_map_zero(self.ball.exp_map_zero(candidate_tangent))
        combined = (alpha[:,None,:,None] * tangent).sum(2)
        return self.ball.distance(query[:,None], self.ball.exp_map_zero(combined))

    def pretraining_loss(self, batch, candidate_batch, candidate_count, tree_distances):
        _, summaries, tangent = self.encode(batch)
        query, alpha = self.query(summaries, tangent, batch["observed"])
        _, _, raw_candidates = self.encode(candidate_batch)
        candidates = raw_candidates.reshape(query.shape[0], candidate_count, 3, -1)
        distances = self.candidate_distances(query, alpha, candidates)
        target = (-tree_distances / self.cfg["tau_h"]).softmax(-1)
        log_prob = (-distances / self.cfg["tau_r"]).log_softmax(-1)
        return -(target * log_prob).sum(-1).mean()

    @torch.no_grad()
    def retrieve(self, batch, bank):
        sequences, summaries, tangent = self.encode(batch)
        query, alpha = self.query(summaries, tangent, batch["observed"])
        k = self.cfg["top_k"]
        if bank["coordinates"].shape[0] < k:
            raise ValueError("Memory contains fewer entries than top_k")
        best_dist, best_idx = None, None
        n = len(bank["records"])
        for start in range(0, n, self.cfg["memory_chunk_size"]):
            points = bank["coordinates"][start:start+self.cfg["memory_chunk_size"]].to(query.device)
            z = self.ball.log_map_zero(points)
            combined = (alpha[:,None,:,None] * z[None]).sum(2)
            distances = self.ball.distance(query[:,None], self.ball.exp_map_zero(combined))
            indices = torch.arange(start, start+len(points), device=query.device)[None].expand(query.shape[0],-1)
            if best_dist is not None:
                distances = torch.cat([best_dist, distances], 1)
                indices = torch.cat([best_idx, indices], 1)
            best_dist, order = distances.topk(min(k, distances.shape[1]), largest=False, sorted=True)
            best_idx = indices.gather(1, order)
        selected = best_idx.cpu()
        evidence = {m:bank["evidence"][m][selected].to(query.device) for m in ("audio", "visual")}
        return {"sequences": sequences, "summaries": summaries, "distances": best_dist,
                "indices": selected, "evidence": evidence, "alpha": alpha}
