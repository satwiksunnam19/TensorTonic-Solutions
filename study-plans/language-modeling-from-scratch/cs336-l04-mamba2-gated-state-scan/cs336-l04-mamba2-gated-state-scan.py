import torch

def mamba2_gated_scan(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, gamma: torch.Tensor) -> dict:
    """
    Returns a dict of tensors: outputs and final_state.
    """
    # pass
    # q=torch.tensor(q)
    # k=torch.tensor(k)
    # v=torch.tensor(v)
    # gamma=torch.tensor(gamma)
    B,s_q,Dk=q.shape 
    Dv=v.shape[-1]
    ys=[]
    gamma = gamma.to(dtype=q.dtype, device=q.device)
    S=torch.zeros(B,Dk,Dv,dtype=q.dtype,device=q.device)
    for t in range(s_q):
        q_t,k_t,v_t=q[:,t],k[:,t],v[:,t]
        g_t = gamma[:, t].view(B, 1, 1) 
        S=g_t*S+torch.unsqueeze(k_t,-1)*torch.unsqueeze(v_t,1)
        y = torch.bmm(q_t.unsqueeze(1), S).squeeze(1)
        ys.append(y)
        print(y.shape)
    recurrent_output=torch.stack(ys,dim=1)
    final_state=S 
    return {"outputs": recurrent_output,"final_state": final_state}

# print(mamba2_gated_scan( q = [[[1,0],[0,1]]], k = [[[1,0],[0,1]]], v = [[[2,3],[4,5]]], gamma = [[0.5,1]]))