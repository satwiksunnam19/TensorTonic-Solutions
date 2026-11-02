import torch
import math 

def causal_gqa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Returns the attention tensor with the same shape and dtype as q.
    """
    # we need to understand few things in here 
    # q is (batch,H_q,seq,head_width)
    # k is (batch,H_kv,seq,head_width)
    # v is (batch,H_kv,seq, head_width)
    # q=torch.tensor(q)
    # k=torch.tensor(k)
    # v=torch.tensor(v)
    h_q=q.shape[1]
    h_kv=k.shape[1]
    m_b_shape=q.shape[2]
    # print(q.shape)
    G=h_q//h_kv # how may groups we need to split 
    # kv_idx=h_q//h_kv # what's this use ? , it need to be computed per query head , which means looping 
    out=torch.empty_like(q)
    print(out.shape)
    for h in range(h_q):
        kv_idx=h//G
        q_h=q[:,h,:,:] 
        # The use of this is to know which kv head need to use the which query head, beacuse the slots are more in the q compared to the kv 
        k_h=k[:,kv_idx,:,:]
        v_h=v[:,kv_idx,:,:]
        M = torch.triu(
    torch.full((m_b_shape,m_b_shape), float("-inf"), dtype=q.dtype, device=q.device),
    diagonal=1,)       
        scale=math.sqrt(q.shape[-1])
        k_h=k_h.transpose(-2,-1)
        internal_math = (q_h @ k_h / scale) + M
        weights = torch.softmax(internal_math, dim=-1)
        out[:, h] = weights @ v_h
    return out 
