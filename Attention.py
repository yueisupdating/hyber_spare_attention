import torch
from torch import nn
import math
from typing import Callable, Optional, Union, TypedDict
from typing_extensions import Unpack

from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.cache_utils import Cache

from flash_attn import flash_attn_func

# 超参数
sink_num=4
block_size=8
local_blocks=4
topk_num=8


class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the forward pass of a `PreTrainedModel`.

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
    """

    num_items_in_batch: Optional["torch.Tensor"]
    output_hidden_states: Optional[bool]
    output_attentions: Optional[bool]
    output_router_logits: Optional[bool]
    cu_seq_lens_q: Optional["torch.LongTensor"]
    cu_seq_lens_k: Optional["torch.LongTensor"]
    max_length_q: Optional[int]
    max_length_k: Optional[int]

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor: # 把头数从kv heads->q heads
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

def rotate_half(x):
    """若原向量表示为 [a, b]，旋转后得到 [-b, a]，这与复数 (a + ib) 乘以 i 后结果 (−b + ia) 的实虚部分排列一致。"""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim) # cos,sin从(b,s,d)->(b,1,s,d)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin) # rope(x)是 x*(cos+i*sin)
    return q_embed, k_embed

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    def __init__(self, config,layer_idx: int):
        super().__init__()

        self.layer_idx = layer_idx
        self.head_dim =config['hidden_size'] // config['num_attention_heads']
        self.num_key_value_groups = config['num_attention_heads'] // config['num_key_value_heads']
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config['attention_dropout']
        self.is_causal = True
        self.config=config
        
        self.q_proj = nn.Linear(
            config['hidden_size'], config['num_attention_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.k_proj = nn.Linear(
            config['hidden_size'], config['num_key_value_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.v_proj = nn.Linear(
            config['hidden_size'], config['num_key_value_heads'] * self.head_dim, bias=config['attention_bias']
        )
        self.o_proj = nn.Linear(
            config['num_attention_heads'] * self.head_dim, config['hidden_size'], bias=config['attention_bias']
        )
    
    def sink_attention(self,query,key,value,attention_mask,scaling,dropout=0.0):
        key_states = repeat_kv(key, self.num_key_value_groups)
        value_states = repeat_kv(value, self.num_key_value_groups)
        bs,h,s=query.shape[:-1]
        mask=torch.zeros((bs,h,s,key_states.shape[-2]),dtype=torch.bool)
        for i in range(query.shape[-2]):
            mask[:,:,i,:min(sink_num,i)]=True
            local_start=max(0,i - local_blocks * block_size)
            mask[:,:,i,local_start:i]=True
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling # (b,h,s1,d) * (b,h,d,s2)-> (b,h,s1,s2)
        attn_weights[~mask]=float("-inf")
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(query.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states) # b,h,s,d
        attn_output = attn_output.contiguous().view(query.shape[0], query.shape[2], -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def dsa_static(self,q,k,v,attention_mask,scaling,dropout=0.0):
        k= repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        seq_len=q.shape[-2]
        num_block=(seq_len+block_size-1)//block_size
        # 计算key block平均表征
        key_block=[]
        for i in range(num_block):
            block_start=i*block_size
            block_end=min(block_start+block_size,seq_len)
            key_block.append(k[:,:,block_start:block_end,:].mean(dim=-2,keepdim=True)) # b,h,s,d
        key_block=torch.cat(key_block,dim=-2).to(torch.float16) # (b,h,s,d) * (b,h,blocks,d)
        qk_block=torch.matmul(q, key_block.transpose(2, 3)) # b,h,s,blocks
        _,topk_idx=torch.topk(qk_block,k=min(topk_num,num_block),dim=-1,largest=True, sorted=False)
        mask=torch.zeros_like(qk_block,dtype=torch.bool)
        for i in range(num_block):
            mask[:,:,i*block_size:(i+1)*block_size ,i] = True
        mask.scatter_(dim=-1, index=topk_idx, value=True)
        mask=mask.repeat_interleave(block_size,dim=-1)
        mask = mask[..., :seq_len] # 将mask重复block_size次，最后维度由num_block->seq_len.为避免seq_len超出block_size的倍数，需要截断
        mask=torch.logical_and(mask,torch.tril(mask))
        
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * scaling # (b,h,s1,d) * (b,h,d,s2)-> (b,h,s1,s2)
        attn_weights[~mask]=float("-inf")
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, v) # b,h,s,d
        attn_output = attn_output.view(q.shape[0], q.shape[2], -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2) # b,h,s,d

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # print(query_states.shape,key_states.shape)

        if self.layer_idx>=math.ceil(0.2*self.config['num_hidden_layers']) and self.layer_idx<=math.ceil(0.85*self.config['num_hidden_layers']):
            return self.sink_attention(query_states,key_states,value_states,attention_mask,scaling=self.scaling,dropout=0.0 if not self.training else self.attention_dropout)
        
        elif self.layer_idx>math.ceil(0.85*self.config['num_hidden_layers']): # b,h,s,d
            if query_states.shape[-2]==key_states.shape[-2]:
                return self.dsa_static(query_states,key_states,value_states,attention_mask,self.scaling,dropout=0.0 if not self.training else self.attention_dropout)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            query_states=query_states.transpose(1, 2)
            key_states=key_states.transpose(1, 2)
            value_states=value_states.transpose(1, 2)
            att_out=flash_attn_func(query_states,key_states,value_states,softmax_scale=self.scaling,causal=True)
            attn_output = att_out.contiguous().view(*att_out.shape[:-2], -1)
            att_output=self.o_proj(attn_output)
            return att_output,None

