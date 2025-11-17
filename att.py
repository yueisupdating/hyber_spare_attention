import torch
from torch import nn
import torch.nn.functional as F
from flash_attn import flash_attn_func

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)

block_size = 32
sink_num = 4
local_blocks = 2
topk_num = 3

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 128256
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, layer_idx: int,args=ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.layer_idx = layer_idx
        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        ).cuda()

    def layer2mask(layer_idx: int,xq: torch.Tensor,keys: torch.Tensor,values: torch.Tensor):
        def att_dsa_dynamic(q,k,v,h,s,cache_s,d):
            d_d=d//2
            W_q_d = 0.01 * torch.randn((d, h, d_d), device=q.device, dtype=torch.float32)
            W_k_d = 0.01 * torch.randn((d, h, d_d), device=q.device, dtype=torch.float32)
            num_block = (s + block_size - 1) // block_size
                
            key_block_lora = []
            for i in range(num_block):
                block_start = i * block_size
                block_end = min(block_start + block_size, s)
                k_block = k[:,block_start:block_end,:,:]  # (bs,block_size, h, d)
                k_d = torch.einsum("bshd,bdhD->bshD", k_block, W_k_d)  # (bs,block_size, h, d_d)
                # 对block内的tokens进行平均池化得到block表示
                key_block_lora.append(k_d.mean(dim=1))  # (bs,h, d_d)
            
            key_block_lora = torch.stack(key_block_lora, dim=1).to(torch.float32)  # (bs,num_block, h, d_d)
            
            # 计算query的压缩表示
            q_d = torch.einsum("bshd,bdhD->bshD", q, W_q_d)  # (bs,seq_len, h, d_d)
            qk_d = torch.einsum("bxhD,byhD->bhxy", q_d, key_block_lora)  # (bs,h, seq_len, num_block)
            _, topk_idx = torch.topk(qk_d, k=min(topk_num, num_block), dim=-1, largest=True, sorted=False)
            
            mask = torch.zeros_like(qk_d, dtype=torch.bool) 
            for i in range(s):
                block_idx = (i+block_size-1) // block_size
                mask[:,:, i, block_idx] = True
            mask.scatter_(-1, topk_idx, True)
            mask = mask.repeat_interleave(block_size, dim=-1)[:,:, :, :s]
            mask = torch.logical_and(mask, torch.tril(mask))
            return mask

        def att_dsa_static(q,k,h,s,cache_s):
            num_block=(s+block_size-1)//block_size
            key_block=[]
            for i in range(num_block):
                block_start=i*block_size
                block_end=min(block_start+block_size,s)
                key_block.append(k[:,block_start:block_end,:,:].mean(dim=1)) # k: bs,cache_len+seqlen,h,d
            key_block=torch.cat(key_block,dim=1).to(torch.float32) #(bs,num_block,h,d)
            qk_gate = torch.einsum("bxhd,byhd->bhxy", q, key_block) #(bs,h,s,num_block)
            topk_val,topk_idx = torch.topk(qk_gate,k=min(topk_num,num_block),dim=-1, largest=True, sorted=False)
            mask = torch.zeros_like(qk_gate, dtype=torch.bool) 
            # 为每个query位置设置对应的block mask
            for i in range(s):
                block_idx = (i+block_size-1) // block_size
                mask[:, i, block_idx] = True
            mask.scatter_(-1, topk_idx, True)
            mask=mask.repeat_interleave(block_size,dim=-1)[:,:,:s] # 将mask重复block_size次，最后维度由num_block->seq_len.为避免seq_len超出block_size的倍数，需要截断
            mask=torch.logical_and(mask,torch.tril(mask))
            return mask

        if layer_idx<=6:
            return None
        else:
            bs,s,h,_=xq.shape
            cache_s=keys.shape[1]
              
            if layer_idx>27:
                return att_dsa_static(xq,keys,values,h,s,cache_s)
            else:
                mask = torch.zeros(bs, h, s, cache_s + s, dtype=torch.bool) # bs,h,s,cache+s
                for i in range(s):
                    mask[:,:,i,:min(sink_num,i)]=True
                    local_start = max(0, i - local_blocks * block_size)
                    mask[:,:,i,local_start:i]=True
                return mask
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(
            keys, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(
            values, self.n_rep
        )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        if keys.shape[1]==seqlen:

            mask=self.layer2mask(self.layer_idx,xq,keys,values) # bs s h d
            scores =  torch.einsum("bxhd,byhd->bhxy", xq, keys)/ math.sqrt(self.head_dim) 
            # scores: bs,h,seqlen,cache_len+seqlen
            # values:bs,cache_len+seqlen,h,d
            if mask is not None:
                scores[~mask] = float("-inf")  # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = torch.einsum("bhxy,byhd->bxhd", scores, values)  # (bs, seqlen,n_local_heads,head_dim)
            output = output.contiguous().view(bsz, seqlen, -1)
            return self.wo(output)
        else:
            dropout=0.0
            scaling=None
            return flash_attn_func(xq, keys, values, dropout, scaling, True)


