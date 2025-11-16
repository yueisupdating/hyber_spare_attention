import torch

block_size = 32
sink_num = 4
local_blocks = 2
topk_num_static = 64

def att_full(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int
    ):
    bs=cu_seqlens.numel()-1
    d_k=torch.rsqrt(q.shape[-1])
    o = torch.zeros_like(q)
    for b in range(bs):
        b_start=cu_seqlens[b].item()
        b_end=cu_seqlens[b+1].item()
        q_=q[b_start:b_end].to(torch.float32)
        k_=k[b_start:b_end].to(torch.float32)
        v_=v[b_start:b_end].to(torch.float32)
        
        qk = torch.einsum("xhd,yhd->hxy", q_, k_)
        qk *= d_k
        p = qk.softmax(dim=-1)
        out = torch.einsum("hxy,yhd->xhd", p, v_)
        o[b_start:b_end] = out.to(q.dtype)
    return o


def att_sink(q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int):
    bs=cu_seqlens.numel()-1
    d_k=torch.rsqrt(q.shape[-1])
    o = torch.zeros_like(q)
    for b in range(bs):
        b_start=cu_seqlens[b].item()
        b_end=cu_seqlens[b+1].item()
        q_=q[b_start:b_end].to(torch.float32)
        k_=k[b_start:b_end].to(torch.float32)
        v_=v[b_start:b_end].to(torch.float32)
        qk = torch.einsum("xhd,yhd->hxy", q_, k_)
        mask = torch.zeros_like(qk, dtype=torch.bool)
        seq_len = b_end - b_start
        for i in range(seq_len):
            mask[:,i,:sink_num]=True
            local_start = max(0, i - local_blocks * block_size)
            mask[:,i,local_start:i]=True
        qk[~mask] = float("-inf")
        qk *= d_k
        p = qk.softmax(dim=-1)
        out = torch.einsum("hxy,yhd->xhd", p, v_)
        o[b_start:b_end] = out.to(q.dtype)
    return o

def att_dsa_static(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int
):
    bs=cu_seqlens.numel()-1
    d_k=torch.rsqrt(torch.tensor(q.shape[-1], dtype=torch.float32))
    o = torch.zeros_like(q)
    for b in range(bs):
        b_start=cu_seqlens[b].item()
        b_end=cu_seqlens[b+1].item()
        q_=q[b_start:b_end].to(torch.float32)
        k_=k[b_start:b_end].to(torch.float32)
        v_=v[b_start:b_end].to(torch.float32)
        seq_len = b_end - b_start
        num_block=(seq_len+block_size-1)//block_size
        key_block=[]
        for i in range(num_block):
            block_start=i*block_size
            block_end=min(block_start+block_size,seq_len)
            key_block.append(k_[block_start:block_end].mean(dim=0))
        key_block=torch.cat(key_block,dim=0).to(torch.float32) #(num_block,h,d)
        qk_gate = torch.einsum("xhd,yhd->hxy", q_, key_block) #(h,s,num_block)
        topk_val,topk_idx = torch.topk(qk_gate,k=min(topk_num_static,num_block),dim=-1, largest=True, sorted=False)
        mask = torch.zeros_like(qk_gate, dtype=torch.bool)
        # 为每个query位置设置对应的block mask
        for i in range(seq_len):
            block_idx = (i+block_size-1) // block_size
            mask[:, i, block_idx] = True
        mask.scatter_(-1, topk_idx, True)
        mask=mask.repeat_interleave(block_size,dim=-1)[:,:,:seq_len] # 将mask重复block_size次，最后维度由num_block->seq_len.为避免seq_len超出block_size的倍数，需要截断
        mask=torch.logical_and(mask,torch.tril(mask))
        qk=torch.einsum("xhd,yhd->hxy", q_, k_)
        qk[~mask] = float("-inf")
        qk *= d_k
        p = qk.softmax(dim=-1)
        out = torch.einsum("hxy,yhd->xhd", p, v_)
        o[b_start:b_end] = out.to(q.dtype)
    return o

def att_dsa_dynamic(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int
):
    bs=cu_seqlens.numel()-1
    head_num,d=q.shape[-2],q.shape[-1]
    d_d=d//2
    d_k=torch.rsqrt(d)
    o = torch.zeros_like(q)
        
    W_q_d = 0.01 * torch.randn((d, head_num, d_d), device=q.device, dtype=torch.float32)
    W_k_d = 0.01 * torch.randn((d, head_num, d_d), device=q.device, dtype=torch.float32)
    
    for b in range(bs):
        b_start=cu_seqlens[b].item()
        b_end=cu_seqlens[b+1].item()
        q_=q[b_start:b_end].to(torch.float32)
        k_=k[b_start:b_end].to(torch.float32)
        v_=v[b_start:b_end].to(torch.float32)
        
        seq_len = b_end - b_start
        num_block = (seq_len + block_size - 1) // block_size
        
        key_block_lora = []
        for i in range(num_block):
            block_start = i * block_size
            block_end = min(block_start + block_size, seq_len)
            k_block = k_[block_start:block_end]  # (block_size, h, d)
            k_d = torch.einsum("shd,dhD->shD", k_block, W_k_d)  # (block_size, h, d_d)
            # 对block内的tokens进行平均池化得到block表示
            key_block_lora.append(k_d.mean(dim=0))  # (h, d_d)
        
        key_block_lora = torch.stack(key_block_lora, dim=0).to(torch.float32)  # (num_block, h, d_d)
        
        # 计算query的压缩表示
        q_d = torch.einsum("shd,dhD->shD", q_, W_q_d)  # (seq_len, h, d_d)
        qk_d = torch.einsum("xhD,yhD->hxy", q_d, key_block_lora)  # (h, seq_len, num_block)
        _, topk_idx = torch.topk(qk_d, k=min(topk_num_static, num_block), dim=-1, largest=True, sorted=False)
        
        mask = torch.zeros((qk_d.shape[0], seq_len, num_block), dtype=torch.bool, device=qk_d.device)
        for i in range(seq_len):
            block_idx = (i+block_size-1) // block_size
            mask[:, i, block_idx] = True
        mask.scatter_(-1, topk_idx, True)
        mask = mask.repeat_interleave(block_size, dim=-1)[:, :, :seq_len]
        mask = torch.logical_and(mask, torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=mask.device)))
        
        qk = torch.einsum("xhd,yhd->hxy", q_, k_)  # (h, seq_len, seq_len)
        qk[~mask] = float("-inf")
        qk *= d_k
        p = qk.softmax(dim=-1)
        out = torch.einsum("hxy,yhd->xhd", p, v_)
        o[b_start:b_end] = out.to(q.dtype)
    return o
