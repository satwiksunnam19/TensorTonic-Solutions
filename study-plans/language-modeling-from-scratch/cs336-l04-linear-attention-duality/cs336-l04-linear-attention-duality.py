import torch

def linear_attention_duality(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: parallel_output, recurrent_output, final_state.
    """
    # pass
    # parallel 
    scores=torch.bmm(q,k.transpose(1,2))
    masked_cores=torch.tril(scores)
    parallel=torch.bmm(masked_cores,v)

    # recurrent 
    B,s_q,D_k=q.shape
    D_v=v.shape[-1]
    state=torch.zeros(B,D_k,D_v,dtype=q.dtype,device=q.device)
    ys=[]
    for t in range(s_q):
        k_t, v_t, q_t = k[:, t], v[:, t], q[:, t]
        state = state + torch.unsqueeze(k_t, -1) * torch.unsqueeze(v_t, 1)
        y = torch.bmm(q_t.unsqueeze(1), state).squeeze(1)
        ys.append(y)
    recurrent_output = torch.stack(ys, dim=1)   # (B, Sq, Dv)
    final_state = state                          # (B, Dk, Dv)

    return {"parallel_output": parallel,"recurrent_output": recurrent_output,"final_state": final_state}