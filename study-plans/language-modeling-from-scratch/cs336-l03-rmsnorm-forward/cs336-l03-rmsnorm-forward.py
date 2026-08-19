import torch

def rmsnorm(x, g, epsilon):
    """
    Returns: RMS-normalized tensor
    """
    sum_=[]
    # x=torch.tensor(x,dtype=torch.float32)
    # g=torch.tensor(g)
    # print(g.shape,x.shape)
    mean= (x**2).mean(dim=-1, keepdim=True) 
    sqrd= torch.sqrt(mean+epsilon) 
    normalized=x/sqrd
    final= torch.mul(normalized,g)
    return final 
