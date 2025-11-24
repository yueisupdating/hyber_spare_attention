import os
import json
import argparse
import math
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from Attention import LlamaAttention

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend="nccl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/models/Llama-3.2-3B-Instruct")
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,dtype=torch.float16)
    config_path=args.model+'/config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        if layer_idx < math.ceil(0.2*config['num_hidden_layers']):
            continue
        layer.self_attn=LlamaAttention(config,layer_idx)
    
    model=model.to(device, dtype=torch.float16)
    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device).to(device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    if local_rank == 0:
        print(tokenizer.decode(tokens.squeeze().tolist()))