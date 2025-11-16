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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/Qwen2.5-7B-Instruct")
    args=parser.parse_args()
    register_att()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)

    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    print(tokens)
    print(tokenizer.decode(tokens.squeeze().tolist()))