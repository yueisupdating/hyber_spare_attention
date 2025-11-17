import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from typing import Callable, Tuple, Optional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import argparse
import torch.distributed as dist
from fairscale.nn.model_parallel.initialize import initialize_model_parallel, model_parallel_is_initialized

from att import Attention

dist.init_process_group(backend="nccl")

if not model_parallel_is_initialized():
    model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))  # 或者您指定的并行大小
    initialize_model_parallel(model_parallel_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/Meta-Llama-3.1-8B-Instruct")
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,attn_implementation="flash_attention_2").to(device)
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        layer.self_attn.attn_fn=Attention(layer_idx=layer_idx)

    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device).to(device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    print(tokenizer.decode(tokens.squeeze().tolist()))