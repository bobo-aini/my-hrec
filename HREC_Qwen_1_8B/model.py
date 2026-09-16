"""HREC integration for the bundled Qwen-1.8B architecture."""
import torch
from hrec.model import HREC
from .backbone.modeling_qwen import QWenLMHeadModel
from .backbone.tokenization_qwen import QWenTokenizer


def load_backbone(cfg):
    options = {"local_files_only": cfg["local_files_only"]}
    if cfg.get("llm_revision"):
        options["revision"] = cfg["llm_revision"]
    dtype = getattr(torch, cfg["llm_dtype"])
    tokenizer = QWenTokenizer.from_pretrained(cfg["llm_path"], **options)
    full = QWenLMHeadModel.from_pretrained(
        cfg["llm_path"], torch_dtype=dtype, use_flash_attn=False,
        fp16=dtype == torch.float16, bf16=dtype == torch.bfloat16,
        fp32=dtype == torch.float32, **options)
    return full.transformer, tokenizer


def hidden_states(model, inputs, mask):
    return model(inputs_embeds=inputs, attention_mask=mask.long(),
                 return_dict=True, use_cache=False).last_hidden_state


class HRECQwen(HREC):
    """Shared HREC stages with Qwen rotary attention and its native tokenizer."""
    def __init__(self, cfg, retriever, bank, backbone=None):
        if cfg["llm_family"] != "qwen":
            raise ValueError("HRECQwen requires llm_family=qwen")
        super().__init__(cfg, retriever, bank, backbone)
