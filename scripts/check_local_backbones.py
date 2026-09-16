"""Validate local checkpoint loading and HREC training for reduced random models.

Creates temporary SentencePiece assets and randomly initialized ChatGLM/Llama
checkpoints. No external model weights or network access are required.
"""
import argparse
import gc
import json
import shutil
import subprocess
import sys
from pathlib import Path

import sentencepiece as spm
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hrec.config import load_config
from hrec.demo import create_demo
from hrec.engine import save_json, restore_task
from hrec.data import FeatureDataset,collate_features,to_device,apply_pattern


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",required=True)
    parser.add_argument("--output",required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True,exist_ok=False)
    corpus = directory/"tokenizer_training.txt"
    corpus.write_text(("A speaker describes an event. Predict emotion. Multimodal evidence. "
                       "Caption: A speaker reacts. Emotion: neutral joy sadness. Distance: 0.123456789\n")*20)
    spm.SentencePieceTrainer.train(input=str(corpus),model_prefix=str(directory/"fixture"),
                                   vocab_size=80,model_type="unigram",character_coverage=1.0,
                                   hard_vocab_limit=False,minloglevel=2)
    results = {}
    for family, package in (("chatglm","HREC_ChatGLM3_6B"),("llama","HREC_Llama2_7B")):
        torch.manual_seed(23)
        checkpoint_dir = directory/f"{family}_random_checkpoint"
        checkpoint_dir.mkdir()
        shutil.copy2(directory/"fixture.model",checkpoint_dir/"tokenizer.model")
        if family == "chatglm":
            from HREC_ChatGLM3_6B.backbone.configuration_chatglm import ChatGLMConfig
            from HREC_ChatGLM3_6B.backbone.modeling_chatglm import ChatGLMForConditionalGeneration
            from HREC_ChatGLM3_6B.backbone.tokenization_chatglm import ChatGLMTokenizer
            config = ChatGLMConfig(num_layers=2,padded_vocab_size=256,hidden_size=32,
                                    ffn_hidden_size=64,kv_channels=8,num_attention_heads=4,
                                    seq_length=512,torch_dtype=torch.float32,
                                    multi_query_attention=True,multi_query_group_num=2)
            model = ChatGLMForConditionalGeneration(config,empty_init=False)
            tokenizer = ChatGLMTokenizer(vocab_file=str(checkpoint_dir/"tokenizer.model"))
        else:
            from HREC_Llama2_7B.backbone.configuration_llama import LlamaConfig
            from HREC_Llama2_7B.backbone.modeling_llama import LlamaForCausalLM
            from HREC_Llama2_7B.backbone.tokenization_llama import LlamaTokenizer
            config = LlamaConfig(vocab_size=256,hidden_size=32,intermediate_size=64,
                                  num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=4,
                                  max_position_embeddings=512)
            model = LlamaForCausalLM(config)
            tokenizer = LlamaTokenizer(vocab_file=str(checkpoint_dir/"tokenizer.model"),legacy=True)
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.ndim == 1:
                    parameter.fill_(1)
                else:
                    parameter.normal_(0,.02)
        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)
        del model,tokenizer
        gc.collect()
        config_path = create_demo(directory/family)
        cfg = load_config(config_path)
        cfg.update({"llm_backend":"huggingface","llm_family":family,
                    "llm_path":str(checkpoint_dir),"llm_dtype":"float32"})
        save_json(config_path,cfg)
        # Use each independent model entry point, not the shared fixture backend.
        command = [sys.executable,"-m",package+".run","pipeline","--config",str(config_path)]
        subprocess.run(command,check=True)
        for pattern in ("no_audio","no_visual","no_text","text_only"):
            subprocess.run([sys.executable,"-m",package+".run","evaluate","--config",str(config_path),"--pattern",pattern],check=True)
        hrec = restore_task(cfg)
        dataset = FeatureDataset(cfg["test_manifest"],cfg)
        batch = apply_pattern(to_device(collate_features([dataset[0],dataset[1]]),"cpu"),"all")
        with torch.no_grad():
            reference = hrec(batch)["prediction"]
            for row in batch["records"]:
                row.pop("label",None)
                row.pop("hierarchy",None)
            torch.testing.assert_close(reference,hrec(batch)["prediction"])
        run = Path(cfg["run_dir"])
        training = json.loads((run/"training_report.json").read_text())
        assert training["model_class"].startswith(package+".")
        assert training["backbone_class"].startswith(package+".backbone.")
        assert training["completion_changed"] and training["retriever_unchanged"]
        results[family] = {"status":"passed", "scope":"reduced random architecture; synthetic features",
                           "local_checkpoint_and_tokenizer_loading":True,
                           "two_stage_pipeline":True,"five_observation_patterns":True,
                           "checkpoint_reload":True,"target_label_independence":True,
                           "training":training}
        del hrec
        gc.collect()
    save_json(args.output,results)
    print("LOCAL_BACKBONE_PIPELINES_PASSED",flush=True)


if __name__ == "__main__":
    main()
