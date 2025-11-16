import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from typing import Callable, Tuple, Optional
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
import argparse

from att import att_sink, att_dsa_static, att_dsa_dynamic

def hf_to_fa(x: torch.Tensor):
    """
    Args:
        x (torch.Tensor): [batch, heads, seqlen, head_dim]

    Returns:
        torch.Tensor: [batch * seqlen, heads, head_dim]
    """
    return x.permute(0, 2, 1, 3).reshape(-1, x.shape[1], x.shape[3])

def att_layer(
    att_impl: Callable,
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *args,
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, None]:
    batch, q_heads, q_len, head_dim = query.shape
    _, kv_heads, kv_len, _ = key.shape
    
    # prefill phase
    query = hf_to_fa(query)
    key = hf_to_fa(key)
    value = hf_to_fa(value) # [batch * seqlen, heads, head_dim]
    kv_replicas = q_heads // kv_heads
    key = torch.repeat_interleave(key, kv_replicas, dim=1)
    value = torch.repeat_interleave(value, kv_replicas, dim=1)
    cu_seqlens_k = torch.cumsum(
        torch.tensor([0] + [kv_len] * batch, device=query.device),
        dim=0,
        dtype=torch.int32,
    )
    out = att_impl(
        q=query,
        k=key,
        v=value,
        cu_seqlens=cu_seqlens_k,
        max_seqlen=kv_len,
    )
    return out,None

def register_att():
    # ALL_ATTENTION_FUNCTIONS["att_full"] = partial(att_full)
    ALL_ATTENTION_FUNCTIONS["att_sink"] = partial(att_layer,att_sink)
    ALL_ATTENTION_FUNCTIONS["att_dsa_static"] = partial(att_layer,att_dsa_static)
    ALL_ATTENTION_FUNCTIONS["att_dsa_dynamic"] = partial(att_layer,att_dsa_dynamic)

def layer2att(layer_idx):
    if layer_idx<=6:
        return None
    elif layer_idx>27:
        return "att_dsa_static"
    else:
        return "att_sink"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/home/yuegengxin/Meta-Llama-3.1-8B-Instruct")
    args=parser.parse_args()
    register_att()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True,attn_implementation="flash_attention_2")
    layers = model.model.layers
    # for layer_idx, layer in enumerate(layers):
    #     # if layer_idx<=5:
    #     #     continue
    #     # att_fn_name = layer2att(layer_idx)
    #     # print(f"层 {layer_idx}: 使用attention函数 {att_fn_name}")
    #     # if hasattr(layer.self_attn, 'attn_fn'):
    #     #     print('\nygx1')
    #     #     layer.self_attn.attn_fn = ALL_ATTENTION_FUNCTIONS[att_fn_name]
    #     # elif hasattr(layer.self_attn, 'attention_fn'):
    #     #     print('\nygx2')
    #     #     layer.self_attn.attention_fn = ALL_ATTENTION_FUNCTIONS[att_fn_name]
    #     print(f'{layer_idx}\n{layer.self_attn._update_causal_mask}')

    inputs = tokenizer.encode("how are you?")
    input_ids = torch.tensor([inputs], device=model.device)
    tokens = model.generate(input_ids, max_length=200, do_sample=False)
    print(tokenizer.decode(tokens.squeeze().tolist()))