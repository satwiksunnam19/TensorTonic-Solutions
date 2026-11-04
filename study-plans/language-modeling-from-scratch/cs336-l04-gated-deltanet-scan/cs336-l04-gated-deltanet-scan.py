import torch

def gated_deltanet_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: outputs and final_state.
    """
    # pass
    b,s,dk=q.shape 
    dv=v.shape[-1]
    I=torch.eye(dk,dk,dtype=q.dtype,device=q.device)
    S=torch.zeros(b,dk,dv,dtype=q.dtype,device=q.device)
    ys=[]
    for t in range(s):
        k_t,q_t,v_t=k[:,t],q[:,t],v[:,t]
        gamma_t=torch.tensor(gamma[:,t].view(b,1,1),dtype=q.dtype,device=q.device)
        beta_t=torch.tensor(beta[:,t].view(b,1,1),dtype=q.dtype, device=q.device)
        k_unsq=k_t.unsqueeze(-1)*k_t.unsqueeze(1)
        v_unsq = k_t.unsqueeze(-1) * v_t.unsqueeze(1) 
        S = gamma_t * torch.bmm(I - beta_t * k_unsq, S) + beta_t * v_unsq
        ys.append(torch.bmm(q_t.unsqueeze(1), S).squeeze(1))

        # ys.append(state)
    return {"outputs": torch.stack(ys, dim=1), "final_state": S}
