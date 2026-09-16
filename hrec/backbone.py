"""Frozen language model with differentiable input-prefix conditioning."""
import torch
from importlib import import_module
from torch import nn


class TinyCausalModel(nn.Module):
    """Random frozen CPU fixture. Never a substitute for a pretrained backbone."""
    def __init__(self, cfg):
        super().__init__()
        dim = cfg["tiny_hidden"]
        self.embedding = nn.Embedding(259, dim)
        self.position = nn.Embedding(cfg["max_prompt_tokens"]+cfg["max_evidence_tokens"]+16, dim)
        layer = nn.TransformerEncoderLayer(dim, cfg["tiny_heads"], 2*dim, dropout=0., batch_first=True)
        self.layers = nn.TransformerEncoder(layer, cfg["tiny_layers"], enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, mask):
        positions = torch.arange(x.shape[1], device=x.device)
        x = x + self.position(positions)[None]
        causal = torch.ones(x.shape[1], x.shape[1], device=x.device, dtype=torch.bool).triu(1)
        return self.norm(self.layers(x, mask=causal, src_key_padding_mask=~mask.bool()))


class FrozenLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.tokenizer = cfg, None
        self.backend = cfg["llm_backend"]
        if self.backend == "tiny":
            with torch.random.fork_rng():
                torch.manual_seed(741)
                self.model = TinyCausalModel(cfg)
            self.hidden_dim = cfg["tiny_hidden"]
            self.pad_id = 0
        elif self.backend == "huggingface":
            packages = {"qwen":"HREC_Qwen_1_8B", "chatglm":"HREC_ChatGLM3_6B", "llama":"HREC_Llama2_7B"}
            if cfg["llm_family"] not in packages:
                raise ValueError("llm_family must be qwen, chatglm, or llama")
            self.adapter = import_module(packages[cfg["llm_family"]]+".model")
            self.model, self.tokenizer = self.adapter.load_backbone(cfg)
            self.hidden_dim = self.model.get_input_embeddings().weight.shape[1]
            self.pad_id = self.tokenizer.pad_token_id
            if self.pad_id is None:
                self.pad_id = self.tokenizer.eos_token_id
            if self.pad_id is None:
                self.pad_id = getattr(self.tokenizer, "eod_id", 0)
        else:
            raise ValueError("llm_backend must be huggingface or tiny")
        self.model.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def embedding_layer(self):
        if self.backend == "tiny":
            return self.model.embedding
        if self.cfg["llm_family"] == "chatglm":
            return self.model.embedding.word_embeddings
        return self.model.get_input_embeddings()

    def token_ids(self, text):
        if self.backend == "tiny":
            return [int(x)+3 for x in text.encode("utf-8")] or [1]
        return self.tokenizer.encode(text, add_special_tokens=False) or [self.pad_id]

    @torch.no_grad()
    def text_features(self, text, max_length=512):
        ids = torch.tensor(self.token_ids(text)[:max_length], device=self.embedding_layer().weight.device)
        return self.embedding_layer()(ids).float().cpu()

    def format_inputs(self, batch, retrieval, bank):
        prompts, evidence = [], []
        per_item = self.cfg["max_evidence_tokens"] // self.cfg["top_k"]
        if per_item < 8:
            raise ValueError("Evidence token budget too small to preserve each retrieved entry")
        for b, record in enumerate(batch["records"]):
            observed_text = record.get("text", "") if bool(batch["observed"][b,2]) else "[missing text]"
            task = "Predict emotion." if self.cfg["task"] == "classification" else "Predict sentiment intensity."
            prompts.append(self.token_ids(f"{task}\nObserved text: {observed_text}\nMultimodal evidence:")[:self.cfg["max_prompt_tokens"]])
            tokens = []
            for index, distance in zip(retrieval["indices"][b].tolist(), retrieval["distances"][b].detach().cpu().tolist()):
                item = bank["records"][index]
                opening = self.token_ids("Caption: ")
                ending = self.token_ids(f"; Emotion: {item['label']}; Distance: {distance:.6f}\n")
                remaining = per_item - len(opening) - len(ending)
                if remaining < 1:
                    raise ValueError("Increase max_evidence_tokens; label/distance fields would be truncated")
                caption = self.token_ids(item["caption"])[:remaining]
                tokens.extend(opening + caption + ending)
            evidence.append(tokens)
        return prompts, evidence

    def forward(self, prefix, prompts, evidence):
        embedding = self.embedding_layer()
        device, dtype = embedding.weight.device, embedding.weight.dtype
        sequences = []
        for b, (prompt, retrieved) in enumerate(zip(prompts, evidence)):
            left = embedding(torch.tensor(prompt, device=device, dtype=torch.long))
            right = embedding(torch.tensor(retrieved, device=device, dtype=torch.long))
            sequences.append(torch.cat([left, prefix[b:b+1].to(dtype), right], dim=0))
        lengths = torch.tensor([len(x) for x in sequences], device=device)
        inputs = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        mask = torch.arange(inputs.shape[1], device=device)[None] < lengths[:,None]
        # No no_grad here: frozen parameters still propagate gradients to prefix.
        if self.backend == "tiny":
            hidden = self.model(inputs, mask)
        else:
            hidden = self.adapter.hidden_states(self.model, inputs, mask)
        if hidden.shape[:2] != inputs.shape[:2]:
            raise RuntimeError(f"Unexpected backbone hidden layout: {tuple(hidden.shape)}")
        return hidden[torch.arange(len(sequences), device=device), lengths-1].float()
