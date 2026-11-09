import torch

def topk_moe_router(x: torch.Tensor, w_router: torch.Tensor, expert_outputs: torch.Tensor, shared_output: torch.Tensor, k: int) -> dict:
    """
    Returns tensors in a dict: output, selected_experts, selected_weights, token_fraction, probability_mass.
    """
    # pass
    T, D = x.shape                       # x: (T, D)
    E = w_router.shape[1]                # w_router: (D, E)
    Dout = expert_outputs.shape[-1]      # expert_outputs: (T, E, Dout)

    # 1. router scores
    scores = x @ w_router                # (T, E)

    # 2. top-k, ties go to the lower expert index
    #    a stable descending sort keeps the lower index first among equal scores
    order = torch.sort(scores, dim=1, descending=True, stable=True).indices  # (T, E)
    selected_experts = order[:, :k]                                          # (T, k) int64
    s_sel = torch.gather(scores, 1, selected_experts)                        # (T, k)

    # 3. weights: softmax over the k selected scores only
    selected_weights = torch.softmax(s_sel, dim=-1)                          # (T, k)

    # 4. gather the chosen experts' outputs
    idx = selected_experts.unsqueeze(-1).expand(-1, -1, Dout)                # (T, k, Dout)
    chosen = torch.gather(expert_outputs, 1, idx)                            # (T, k, Dout)

    # 5. weighted sum over k, plus the shared output
    output = shared_output + (selected_weights.unsqueeze(-1) * chosen).sum(dim=1)  # (T, Dout)

    # 6. token fraction f: 0/1 selection matrix, averaged over tokens
    mask = torch.zeros(T, E, dtype=x.dtype, device=x.device)                 # (T, E)
    mask.scatter_(1, selected_experts, torch.ones_like(selected_weights))    # src (T, k)
    token_fraction = mask.mean(dim=0)                                        # (E,)

    # 7. probability mass m: softmax over ALL experts, averaged over tokens
    probability_mass = torch.softmax(scores, dim=-1).mean(dim=0)             # (E,)

    return {
        "output": output,
        "selected_experts": selected_experts,
        "selected_weights": selected_weights,
        "token_fraction": token_fraction,
        "probability_mass": probability_mass,
    }
