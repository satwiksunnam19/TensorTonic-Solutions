import torch

def kda_recurrence(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, decay_logits: torch.Tensor, write_strength: torch.Tensor, output_gate_logits: torch.Tensor, output_projection: torch.Tensor, initial_state: torch.Tensor, g_min: float = -5.0, eps: float = 1e-6) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing sequence outputs and the final recurrent state.
    """
    # pass
    out=[]
    B,S,H,dk=query.shape 
    dv=value.shape[-1]
    state= initial_state.clone().to(query.dtype)
    if write_strength.dim() == 4:
        write_strength = write_strength.squeeze(-1)    # (B, S, H)
    for t in range(S):
        q_t,k_t,v_t=query[:,t],key[:,t],value[:,t] 
        z_t=decay_logits[:,t]
        beta_t=write_strength[:,t]
        gate_t=output_gate_logits[:,t]
        alpha=torch.exp(g_min*torch.sigmoid(z_t))
        decayed=state*alpha.unsqueeze(-1)
        kk = k_t.unsqueeze(-1) * k_t.unsqueeze(-2) 
        right = (beta_t.unsqueeze(-1).unsqueeze(-1) * kk) @ decayed 
        erased=decayed-right 
        write=beta_t.unsqueeze(-1).unsqueeze(-1)*k_t.unsqueeze(-1)*v_t.unsqueeze(-2)
        state=erased+write 
        o_t = (state.transpose(-2, -1) @ q_t.unsqueeze(-1)).squeeze(-1)
        o_t_sq=torch.square(o_t).mean(dim=-1,keepdim=True)+eps
        normed=o_t/torch.sqrt(o_t_sq)
        out_t = normed * torch.sigmoid(gate_t)
        out.append(out_t)
    out_r=torch.stack(out,dim=1).reshape(B,S,H*dv)@output_projection.transpose(-2, -1)
    return {"outputs": out_r, "final_state": state} 
