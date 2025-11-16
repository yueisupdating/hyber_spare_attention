import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import argparse

from att import att_full, att_sink, att_dsa_static, att_dsa_dynamic

def register_att():
    ALL_ATTENTION_FUNCTIONS["att_full"] = partial(att_full)
    ALL_ATTENTION_FUNCTIONS["att_sink"] = partial(att_sink)
    ALL_ATTENTION_FUNCTIONS["att_dsa_static"] = partial(att_dsa_static)
    ALL_ATTENTION_FUNCTIONS["att_dsa_dynamic"] = partial(att_dsa_dynamic)

def layer2att(layer_idx):
    if layer_idx<=5:
        return "att_full"
    elif layer_idx>23:
        return "att_dsa_static"
    else:
        return "att_sink"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/Qwen2.5-7B-Instruct")
    args=parser.parse_args()
    register_att()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    layers = model.model.layers
    for layer_idx, layer in enumerate(layers):
        att_fn_name = layer2att(layer_idx)
        print(f"层 {layer_idx}: 使用attention函数 {att_fn_name}")
        if hasattr(layer.self_attn, 'attn_fn'):
            print(1)
            layer.self_attn.attn_fn = ALL_ATTENTION_FUNCTIONS[att_fn_name]
        elif hasattr(layer.self_attn, 'attention_fn'):
            print(2)
            layer.self_attn.attention_fn = ALL_ATTENTION_FUNCTIONS[att_fn_name]

    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    print(tokens)
    print(tokenizer.decode(tokens.squeeze().tolist()))