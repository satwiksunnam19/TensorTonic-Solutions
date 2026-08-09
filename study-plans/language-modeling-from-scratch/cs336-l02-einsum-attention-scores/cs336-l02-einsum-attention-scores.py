import torch

def attention_scores(q, k, num_heads):
    B, S_q, D = q.shape
    B, S_k, D = k.shape

    h = num_heads
    d_h = D // h

    q = q.reshape(B, S_q, h, d_h).transpose(1, 2)   # (B, H, S_q, d_h)
    k = k.reshape(B, S_k, h, d_h).transpose(1, 2)   # (B, H, S_k, d_h)

    scores = (q @ k.transpose(2, 3)) / (d_h ** 0.5)  # (B, H, S_q, S_k)
    return scores