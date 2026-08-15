import math 
def memory_accountant(param_shapes, param_bytes_per_element, grad_bytes_per_element,
                       activation_shapes, activation_bytes_per_element,
                       optimizer, optimizer_bytes_per_element):
    """
    Returns: dictionary containing exact parameter, gradient, activation, optimizer, and total bytes
    """
    
    soft_mul_list_grad=[]
    Dict_For_Optimizer = {"SGD":0,"AdaGrad":1,"Adam":2}
    for i in param_shapes: 
        soft_mul= math.prod(i)
        # print(soft_mul) 
        soft_mul_list_grad.append(soft_mul)
    
    soft_mul_list=sum(soft_mul_list_grad)
    parameters=param_bytes_per_element*soft_mul_list
    gradients=soft_mul_list*grad_bytes_per_element
    soft_mul_activ_list=[]
    for i in  activation_shapes: 
        soft_mul_activ= math.prod(i)
        # print(soft_mul) 
        soft_mul_activ_list.append(soft_mul_activ)
    soft_mul_activ_list_2=sum(soft_mul_activ_list)
    activations= soft_mul_activ_list_2*activation_bytes_per_element
    # optim_bytes=[]
    for k, v in Dict_For_Optimizer.items():
        if optimizer.lower()==k.lower():
            # print("kv",k,v)
            value=Dict_For_Optimizer[k]
            # print(value)
            optimizer_bytes=soft_mul_list*value*optimizer_bytes_per_element

    total=optimizer_bytes+parameters+gradients+activations

    return {"parameters":parameters,"gradients":gradients,"activations":activations,"optimizer_state":optimizer_bytes,"total":total}
