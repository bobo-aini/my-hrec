"""HREC integration for the bundled Llama2-7B architecture."""
import torch
from hrec.model import HREC
from .backbone.modeling_llama import LlamaForCausalLM
from .backbone.tokenization_llama import LlamaTokenizer


def load_backbone(cfg):
    options = {"local_files_only": cfg["local_files_only"]}
    if cfg.get("llm_revision"):
        options["revision"] = cfg["llm_revision"]
    tokenizer = LlamaTokenizer.from_pretrained(cfg["llm_path"], **options)
    full = LlamaForCausalLM.from_pretrained(
        cfg["llm_path"], torch_dtype=getattr(torch, cfg["llm_dtype"]),
        attn_implementation="eager", **options)
    return full.model, tokenizer


def hidden_states(model, inputs, mask):
    positions = (mask.long().cumsum(-1) - 1).clamp_min(0)
    return model(inputs_embeds=inputs, attention_mask=mask.long(),
                 position_ids=positions, return_dict=True,
                 use_cache=False).last_hidden_state


class HRECLlama(HREC):
    """Shared HREC stages with Llama attention, positions, and tokenization."""
    def __init__(self, cfg, retriever, bank, backbone=None):
        if cfg["llm_family"] != "llama":
            raise ValueError("HRECLlama requires llm_family=llama")
        super().__init__(cfg, retriever, bank, backbone)
