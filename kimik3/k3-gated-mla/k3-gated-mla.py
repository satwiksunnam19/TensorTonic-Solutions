import math
import torch

def gated_mla(hidden_states: torch.Tensor, query_projection: torch.Tensor, latent_down_projection: torch.Tensor, key_up_projection: torch.Tensor, value_up_projection: torch.Tensor, output_gate_projection: torch.Tensor, output_projection: torch.Tensor, num_heads: int, causal: bool = True) -> dict[str, torch.Tensor]:
    """
    Returns a dictionary containing gated attention output and the latent key-value cache.
    """
    # pass
    B, S, D = hidden_states.shape
    D_h = D // num_heads
    Q = hidden_states @ query_projection.T
    C = hidden_states @ latent_down_projection.T
    K = C @ key_up_projection.T
    V = C @ value_up_projection.T
    Q = Q.reshape(B, S, num_heads, D_h)
    K = K.reshape(B, S, num_heads, D_h)
    V = V.reshape(B, S, num_heads, D_h)
    Q = Q.transpose(1, 2)
    K = K.transpose(1, 2)
    V = V.transpose(1, 2)
    A = Q @ K.transpose(-2, -1) / math.sqrt(D_h)
    if causal: 
        mask=torch.triu(torch.ones(S,S),diagonal=1).bool().to(hidden_states.device)
        A=A.masked_fill(mask,float('-inf'))
    A=torch.softmax(A,dim=-1)@V
    A = A.transpose(1, 2)  # (B, H, S, D_h) → (B, S, H, D_h)
    A = A.contiguous().reshape(B, S, D)  # → (B, S, D)
    gate=torch.sigmoid(hidden_states@output_gate_projection.T)*A
    Y=gate@output_projection.T
    return {"output": Y, "latent_cache": C} 