# Third-party source notices

The shared HREC modules retain the inherited MIT license in `LICENSE`. Bundled backbone sources retain their respective copyright headers and licenses.

| Component | Source | License |
|---|---|---|
| Qwen model, tokenizer, utilities | Source accompanying Qwen-1.8B | `HREC_Qwen_1_8B/backbone/LICENSE` and `NOTICE` |
| ChatGLM model, tokenizer, quantization utility | Existing ChatGLM3 implementation with documented HREC interface changes | `HREC_ChatGLM3_6B/backbone/LICENSE` |
| Llama model and tokenizer | Hugging Face Transformers v4.40.2 | `HREC_Llama2_7B/backbone/LICENSE` and `NOTICE` |

Provider references: [Qwen license](https://huggingface.co/Qwen/Qwen-1_8B/blob/main/LICENSE), [ChatGLM source license](https://github.com/zai-org/ChatGLM3/blob/main/LICENSE), and [Transformers v4.40.2 license](https://github.com/huggingface/transformers/blob/v4.40.2/LICENSE).

Modified files carry modification notices. `docs/BACKBONE_SOURCE_MANIFEST.json` records changes and hashes. Upstream source attributions and utility identifiers retain their original meanings. Pretrained weights and datasets are not redistributed; their licenses and access conditions remain separate from those of the implementation files.
