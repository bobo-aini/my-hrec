"""HREC integration for the bundled ChatGLM3-6B architecture."""
import torch
from hrec.model import HREC
from .backbone.configuration_chatglm import ChatGLMConfig
from .backbone.modeling_chatglm import ChatGLMForConditionalGeneration
from .backbone.tokenization_chatglm import ChatGLMTokenizer


def load_backbone(cfg):
    options = {"local_files_only": cfg["local_files_only"]}
    if cfg.get("llm_revision"):
        options["revision"] = cfg["llm_revision"]
    config = ChatGLMConfig.from_pretrained(cfg["llm_path"], **options)
    if config.pre_seq_len is not None or config.quantization_bit:
        raise ValueError("HREC expects an unquantized base ChatGLM3 checkpoint without prefix tuning")
    tokenizer = ChatGLMTokenizer.from_pretrained(cfg["llm_path"], **options)
    full = ChatGLMForConditionalGeneration.from_pretrained(
        cfg["llm_path"], config=config,
        torch_dtype=getattr(torch, cfg["llm_dtype"]), **options)
    return full.transformer, tokenizer


def hidden_states(model, inputs, mask):
    if inputs.shape[1] > model.seq_length:
        raise ValueError("Input exceeds ChatGLM context length")
    output = model(inputs_embeds=inputs, attention_mask=mask.long(),
                   return_dict=True, use_cache=False)
    return output.last_hidden_state.transpose(0, 1).contiguous()


class HRECChatGLM(HREC):
    """Shared HREC stages with native ChatGLM sequence layout and rotary masks."""
    def __init__(self, cfg, retriever, bank, backbone=None):
        if cfg["llm_family"] != "chatglm":
            raise ValueError("HRECChatGLM requires llm_family=chatglm")
        super().__init__(cfg, retriever, bank, backbone)
