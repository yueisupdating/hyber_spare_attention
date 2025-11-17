import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from typing import Callable, Tuple, Optional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import argparse

from att import Attention

def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/Meta-Llama-3.1-8B-Instruct")
    args=parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,attn_implementation="flash_attention_2")
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        layer.self_attn.attn_fn=Attention(layer_idx=layer_idx)

    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    print(tokenizer.decode(tokens.squeeze().tolist()))