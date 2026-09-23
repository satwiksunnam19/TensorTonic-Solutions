import torch

def rotary_embed(q, k, positions, inv_freq):
    """
    Returns: dictionary containing q_rotated and k_rotated
    """
    q_orig_dtype, q_orig_device = q.dtype, q.device
    k_orig_dtype, k_orig_device = k.dtype, k.device

    # always compute in float32, regardless of input dtype (float32 or bfloat16)
    q32 = q.to(dtype=torch.float32)
    k32 = k.to(dtype=torch.float32)
    positions32 = torch.as_tensor(positions, dtype=torch.float32, device=q_orig_device)
    inv_freq32 = torch.as_tensor(inv_freq, dtype=torch.float32, device=q_orig_device)

    B, H, S, D = q32.shape
    q32 = q32.reshape(B, H, S, D // 2, 2)
    k32 = k32.reshape(B, H, S, D // 2, 2)

    theta = positions32.unsqueeze(-1) * inv_freq32  # (S,D//2) or (B,S,D//2)
    theta = theta.unsqueeze(0).unsqueeze(0) if positions32.dim() == 1 else theta.unsqueeze(1)

    x_even_q, x_odd_q = q32[..., 0], q32[..., 1]
    x_even_k, x_odd_k = k32[..., 0], k32[..., 1]
    cos_t, sin_t = torch.cos(theta), torch.sin(theta)

    x_even_new_q = x_even_q * cos_t - x_odd_q * sin_t
    x_odd_new_q  = x_even_q * sin_t + x_odd_q * cos_t
    x_even_new_k = x_even_k * cos_t - x_odd_k * sin_t
    x_odd_new_k  = x_even_k * sin_t + x_odd_k * cos_t

    q_rotated = torch.stack([x_even_new_q, x_odd_new_q], dim=-1).reshape(B, H, S, D)
    k_rotated = torch.stack([x_even_new_k, x_odd_new_k], dim=-1).reshape(B, H, S, D)

    # cast back to each tensor's OWN original dtype/device
    q_rotated = q_rotated.to(dtype=q_orig_dtype, device=q_orig_device)
    k_rotated = k_rotated.to(dtype=k_orig_dtype, device=k_orig_device)

    return {"q_rotated": q_rotated, "k_rotated": k_rotated}