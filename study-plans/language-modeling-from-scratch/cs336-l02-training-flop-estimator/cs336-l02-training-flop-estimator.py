import math 
def flop_estimator(matmuls, attention_flops=0):
    """
    Returns: dictionary containing exact forward, backward, and total FLOP counts
    """
    prod_matmuls=[]
    sum_matmuls=[]
    for i in matmuls:
        soft_mul=math.prod(i)
        prod_matmuls.append(soft_mul)

    for i in prod_matmuls: 
        sum_=2*i
        sum_matmuls.append(sum_)
    forward_flops=sum(sum_matmuls)+attention_flops
    f_backward=2*forward_flops
    f_total=forward_flops+f_backward

    return {"forward_flops": forward_flops, "backward_flops": f_backward, "total_flops" : f_total}

    
