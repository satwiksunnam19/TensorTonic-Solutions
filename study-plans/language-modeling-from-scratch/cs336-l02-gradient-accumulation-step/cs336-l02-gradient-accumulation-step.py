import torch

def gradient_accumulation_step(param, microbatch_inputs, microbatch_targets, lr):
    """
    Returns: dictionary containing new_param and full_grad tensors
    """
    # pass
    # let's do the math and diffrentation 
    # we've the loss function in the terms of w, we need to derive the gradient using the given formula 
    # Loss= sigma over microbatches and sigma over all rows of common dim of input_matrix & Pred_matrix over the squared((x(m,i)w-y(m,i)))
    # This is a classic squared((input-output)) 

    N = sum(len(i) for i in microbatch_inputs)
    # print(N)

    # param=torch.tensor(param,dtype=torch.float32)
    # microbatch_inputs
    # microbatch_inputs= [torch.tensor(microbatch_inputs, dtype=torch.float32)]
    # microbatch_targets= torch.tensor(microbatch_targets, dtype=torch.float32)

    # w= param().detach().clone().requires_grad_(True)
    param = param.detach().clone().requires_grad_(True)

    # Loop over the inputs and ops 
    for x_m,y_m in zip(microbatch_inputs,microbatch_targets):
        pred= torch.matmul(x_m,param)
        residual= pred-y_m
        print(residual.shape, residual)
        loss_m= sum(residual**2)/N 
        loss_m.backward()
    
    full_grad= param.grad.detach().clone()

    # apply one sgd update 

    param= param - (lr*full_grad).detach()

    return {"new_param": param, "full_grad": full_grad}


