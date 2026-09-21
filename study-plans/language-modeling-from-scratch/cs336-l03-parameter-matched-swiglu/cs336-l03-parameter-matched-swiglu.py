import torch
import torch.nn.functional as F
import math

def parameter_matched_swiglu(x, w_g, w_v, w_o, base_params):
    # x = torch.tensor(x, dtype=torch.float32)
    # w_g = torch.tensor(w_g, dtype=torch.float32)
    # w_v = torch.tensor(w_v, dtype=torch.float32)
    # w_o = torch.tensor(w_o, dtype=torch.float32)

    d = x.shape[-1]
    h_max = w_g.shape[1]

    div = math.floor((base_params / (3 * d)) + 0.5)
    h = int(max(1, min(div, h_max)))

    w_g = w_g[:, :h]
    w_v = w_v[:, :h]
    w_o = w_o[:h, :]

    gate = x @ w_g
    value = x @ w_v
    activated = F.silu(gate)
    hidden = activated * value
    output = hidden @ w_o

    parameter_count = int(3 * h * d)

    return {
        "output": output,
        "hidden_width": h,
        "parameter_count": parameter_count
    }