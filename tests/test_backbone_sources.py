"""Check the shipped transformer implementations, without pretrained weights."""
import importlib.util
import unittest
from importlib import import_module

import torch
from hrec.backbone import FrozenLanguageModel


AVAILABLE = all(importlib.util.find_spec(name) is not None
                for name in ("transformers", "tiktoken", "sentencepiece", "einops"))


@unittest.skipUnless(AVAILABLE, "Install requirements.txt for backbone implementation tests")
class BackboneSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_adapter(self, family, package, body):
        torch.manual_seed(19)
        with torch.no_grad():
            for parameter in body.parameters():
                if parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    parameter.normal_(0, .02)
        body.requires_grad_(False).eval()
        wrapper = FrozenLanguageModel.__new__(FrozenLanguageModel)
        torch.nn.Module.__init__(wrapper)
        wrapper.backend = "huggingface"
        wrapper.cfg = {"llm_family":family}
        wrapper.model = body
        wrapper.adapter = import_module(package+".model")
        self.assertTrue(type(body).__module__.startswith(package+".backbone."))
        prefix = torch.randn(2, 32, requires_grad=True)
        prompts, evidence = [[1,2,3,4], [1,2]], [[3,4,5], [3,4,5,6]]
        hidden = wrapper(prefix, prompts, evidence)
        self.assertEqual(tuple(hidden.shape), (2,32))
        self.assertTrue(torch.isfinite(hidden).all())
        hidden.square().mean().backward()
        self.assertGreater(prefix.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in body.parameters()))
        with torch.no_grad():
            singles = torch.cat([wrapper(prefix[i:i+1],prompts[i:i+1],evidence[i:i+1]) for i in range(2)])
        torch.testing.assert_close(hidden, singles, atol=3e-5, rtol=3e-5)
        wrapper.train()
        self.assertFalse(body.training)

    def test_qwen_shipped_source_prefix_gradient_and_padding(self):
        from HREC_Qwen_1_8B.backbone.configuration_qwen import QWenConfig
        from HREC_Qwen_1_8B.backbone.modeling_qwen import QWenModel
        cfg = QWenConfig(vocab_size=128, hidden_size=32, intermediate_size=128,
                         num_hidden_layers=2, num_attention_heads=4, kv_channels=8,
                         max_position_embeddings=64, seq_length=64, fp32=True,
                         use_flash_attn=False, use_dynamic_ntk=False, use_logn_attn=False)
        self.check_adapter("qwen", "HREC_Qwen_1_8B", QWenModel(cfg))

    def test_llama_shipped_source_prefix_gradient_and_padding(self):
        from HREC_Llama2_7B.backbone.configuration_llama import LlamaConfig
        from HREC_Llama2_7B.backbone.modeling_llama import LlamaModel
        cfg = LlamaConfig(vocab_size=128,hidden_size=32,intermediate_size=64,
                          num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
                          max_position_embeddings=64)
        self.check_adapter("llama", "HREC_Llama2_7B", LlamaModel(cfg))

    def test_chatglm_shipped_source_prefix_gradient_and_padding(self):
        from HREC_ChatGLM3_6B.backbone.configuration_chatglm import ChatGLMConfig
        from HREC_ChatGLM3_6B.backbone.modeling_chatglm import ChatGLMModel
        cfg = ChatGLMConfig(num_layers=2,padded_vocab_size=128,hidden_size=32,
                            ffn_hidden_size=64,kv_channels=8,num_attention_heads=4,
                            seq_length=64,torch_dtype=torch.float32,
                            multi_query_attention=True,multi_query_group_num=2)
        body = ChatGLMModel(cfg,empty_init=False)
        self.check_adapter("chatglm", "HREC_ChatGLM3_6B", body)
        ids = torch.tensor([[1,2,3],[2,3,4]])
        with torch.no_grad():
            ordinary = body(input_ids=ids,use_cache=False).last_hidden_state
            embedded = body(inputs_embeds=body.get_input_embeddings()(ids),use_cache=False).last_hidden_state
        torch.testing.assert_close(ordinary,embedded)


if __name__ == "__main__":
    unittest.main()
