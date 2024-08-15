import torch
import torch.nn as nn

def zero_refine(tensor):
    min_values = torch.min(tensor, tensor.T)
    tensor = tensor - min_values
    tensor.fill_diagonal_(0)
    return tensor

def zero_refine_batch(tensor):
    _, L, _ = tensor.shape
    identity_matrix = torch.eye(L, device=tensor.device).unsqueeze(0)  # Shape: (1, L, L)
    min_values = torch.min(tensor, tensor.transpose(1, 2))
    refined_tensor = tensor - min_values
    refined_tensor = refined_tensor * (1 - identity_matrix)
    return refined_tensor

def infeasibility(a_t, K):
    # input size:  L L
    if a_t.ndim != 2:
        raise ValueError(f"Input tensor must be 2-dimensional, but got {a_t.ndim} dimensions")
    constrains_1 = torch.sum(torch.abs(a_t[a_t < 0])) # greater than 0 constrains 
    dk = torch.sum(a_t, dim=1) - K # lower than K constrains
    constrains_2 = torch.sum(dk[dk > 0])
    infesibility = (constrains_1 + constrains_2) / a_t.numel()
    return infesibility

def minimize_objective_quardratic(D, a_t, xi, K, qfrac):
    var = xi - K + torch.sum((a_t - a_t.T), dim=1)
    return torch.sum(D * a_t) + torch.sum(qfrac[0] * var**2 + qfrac[1] * var + qfrac[2])

def minimize_objective_excess(D, a_t, xi, K):
    return torch.sum(D * a_t) + torch.sum(nn.ReLU()(xi - K + torch.sum((a_t - a_t.T), dim=1)))

def minimize_objective_excess_batch(D, a_t, xi, K, L):
    a_t = a_t.reshape(-1, L[0], L[0])
    a_diff = torch.sum((a_t - a_t.transpose(1,2)), dim=2) # B L
    return (torch.sum(D * a_t, dim=(1, 2)) + torch.sum(nn.ReLU()(xi - K + a_diff), dim=1))  / L[0]**2

def minimize_objective_quardratic_batch(D, a_t, xi, K, L, fqurd):
    a_t = a_t.reshape(-1, L[0], L[0])
    a_diff = torch.sum((a_t - a_t.transpose(1,2)), dim=2) # B L
    var = xi - K + a_diff
    return (torch.sum(D * a_t, dim=(1, 2)) + torch.sum(fqurd[0].item()*var**2 + fqurd[1].item()*var + fqurd[2].item(), dim=1))  / L[0]**2 # B,  normalized by batch size and L^2  

# mock exnt loss and obj cost
#-----------------------------------------------------------
def exnt_loss(a_t, y, K, mu, sigma, D, L, args, stage2=False):
    if stage2:
        return nn.CrossEntropyLoss()(a_t, y) # a_t: action matrix,y: 
    else:
        MSE_loss = nn.MSELoss()
       
        obj_value = minimize_objective_quardratic_exnt_batch(D, a_t,  mu, sigma, K, L)
        if args.calculated_obj:
            y_obj_value = minimize_objective_quardratic_exnt_batch(D, y,  mu, sigma, K, L)
        else:
            y_obj_value = y.double()
        return MSE_loss(obj_value, y_obj_value) 

def minimize_objective_quardratic_exnt(D, a_t,  mu, sigma, K):
    return torch.sum(D * a_t) + torch.sum((mu - K + torch.sum((a_t - a_t.T), dim=1))**2)  + torch.sum(sigma**2)

def minimize_objective_quardratic_exnt_batch(D, a_t,  mu, sigma, K, L):
    a_t = a_t.reshape(-1, L[0], L[0])
    a_diff = torch.sum((a_t - a_t.transpose(1,2)), dim=2) # B L
    return ((torch.sum(D * a_t, dim=(1, 2)) + torch.sum((mu - K + a_diff)**2, dim=1)) + torch.sum(sigma**2, dim=1)) / L[0]**2 
#--------------------------------------------------------

def hybird_loss(a_t, y, K, xi, D, L, args, custom_loss=None, custom_loss_lambda=None, stage2=False, qfrac=None):
    loss = 0
    loss_count = 0
    loss_combo = args.loss_combo if custom_loss is None else custom_loss
    loss_lambda = args.loss_lambda if custom_loss_lambda is None else custom_loss_lambda

    if stage2:
        ce_weight = torch.tensor([0.5,0.5]).to(a_t.device)
        return nn.CrossEntropyLoss(weight=ce_weight)(a_t, y)
        # print(a_t.shape, y.shape)
        # return nn.MSELoss()(a_t, y)
    if loss_combo[0] == "1": # 1 infesibility
        loss_count += 1
        constrains_1 = torch.sum(torch.abs(a_t[a_t < 0])) # greater than 0 constrains 
        dk = torch.sum(a_t.reshape(-1, K.shape[-1], K.shape[-1]), dim=1) - K # lower than K constrains
        constrains_2 = torch.sum(dk[dk > 0])
        infesibility = (constrains_1 + constrains_2) / a_t.numel()
        lambda1 = float(loss_lambda[0])
        # print(infesibility)
        loss += lambda1 * infesibility
    if loss_combo[1] == "1": # 2 diag 0 
        loss_count += 1
        diag_product = torch.abs(torch.sum(a_t.reshape(-1, L[0], L[0]) * a_t.reshape(-1, L[0], L[0]).transpose(1,2)))
        part_loss = torch.sqrt(diag_product) / (diag_product.numel() * K.shape[0])
        lambda2 = float(loss_lambda[1])
        loss += lambda2 * part_loss
    if loss_combo[2] == "1": # 3 a_t CE
        loss_count += 1
        pos_weight = torch.ones([y.shape[1]]).to(a_t.device)
        BCE_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        lambda3 = float(loss_lambda[2])
        loss += lambda3 * BCE_loss(a_t, y)
    if loss_combo[3] == "1": # 4 a_t MSE
        loss_count += 1
        MSE_loss = nn.MSELoss() 
        lambda4 = float(loss_lambda[3])
        loss += lambda4 * MSE_loss(a_t, y)
    if loss_combo[4] == "1": # 5 cost MSE loss
        loss_count += 1
        MSE_loss = nn.MSELoss() 
        if args.g_type == "quadratic":
            predicted_cost = minimize_objective_quardratic_batch(D, a_t, xi, K, L, qfrac[0])
            label_cost = minimize_objective_quardratic_batch(D, y, xi, K, L, qfrac[0])
        elif args.g_type == "excess":
            predicted_cost = minimize_objective_excess_batch(D, a_t, xi, K, L)
            label_cost = minimize_objective_excess_batch(D, y, xi, K, L)
        else:
            print(f"no such g type {args.g_type}")
        lambda5 = float(loss_lambda[4])
        # print(K[0], xi[0], D[0], L[0])
        # print(predicted_cost.mean(), label_cost.mean())
        loss += lambda5 * MSE_loss(predicted_cost, label_cost)
    if loss_combo[5] == "1": # 6 pure obj
        lambda6 = float(loss_lambda[5])
        qfrac = args.qfrac
        obj_cost = lambda6 * torch.mean(minimize_objective_quardratic_batch(D, a_t, xi, K, L, qfrac))
        # print(obj_cost)
        loss += obj_cost
    if loss_combo[6] == "1": # 7 stage 2 energy function loss
        loss_count += 1
        lambda7 = float(loss_lambda[6])
        loss += lambda7 * batch_stage_two_loss(a_t, D, xi, K, L)
    return loss / loss_count

def infeasibility_loss_T(output, batch_labels, T, L, Ks, loss_scalar):
    # this loss function includes 
    # 1. MSE loss of the output and target
    # 2. infesibility score of the output violation
    # 3. a_ij * a_ji should be 0, otherwise will be added to the loss.
    # and the loss_scalar to magnify the non zero values.
    infesibility = torch.tensor(0.0).to(output.device)
    Ks = Ks.to(output.device)
    for t in range(T):
        for i in range(L):
            constrain_test =  torch.sum(output[ :, t, i, :], dim=-1) - Ks[:, i]
            infesibility += torch.sum(constrain_test[constrain_test > 0])
            for j in range(L):
                infesibility += torch.sum(torch.abs(output[:, t, i, j][output[:, t, i, j] < 0]))
    diag_product_sum = torch.abs(torch.sum(output * output.transpose(3,2))) / sum(output.shape)
    output = output.reshape(-1, T *L * L)
    loss1 = infesibility / sum(output.shape)
    loss2 = nn.MSELoss()(output, batch_labels)
    loss3 = diag_product_sum
    # print(loss1, loss2, loss3)
    return  loss1 + loss_scalar * loss2 + loss3



# stage 2 loss ————————————————————————————————————————————————————————————————————
def batch_objective(a_batch, D_batch, xi_batch, K_batch, L):
    a_batch = a_batch.view(-1, L, L)  # Reshape each `a` in the batch
    term1 = torch.sum(D_batch * a_batch, dim=(1, 2))
    
    term2 = torch.zeros(a_batch.size(0), device=a_batch.device)
    for i in range(L):
        term2 += (xi_batch[:, i] - K_batch[:, i] + torch.sum(a_batch[:, i, :] - a_batch[:, :, i], dim=1))**2
    
    return term1 + term2

def batch_constraints_satisfied(a_batch, K_batch, L):
    a_batch = a_batch.view(-1, L, L)
    constraints = [] 
    for i in range(L):
        constraints.append(torch.sum(a_batch[:, i, :], dim=1) <= K_batch[:, i])
    constraints_satisfied = torch.stack(constraints, dim=1).all(dim=1)
    non_negative = torch.all(a_batch >= 0, dim=(1, 2))
    
    return constraints_satisfied & non_negative

def batch_energy_function(a_batch, D_batch, xi_batch, K_batch, L):
    constraints_met = batch_constraints_satisfied(a_batch, K_batch, L)
    energy = batch_objective(a_batch, D_batch, xi_batch, K_batch, L)
    energy[~constraints_met] = torch.inf  # Set energy to infinity where constraints are not met
    return energy

# Energy loss function for single output.
def batch_stage_two_loss(a_batch, D_batch, xi_batch, K_batch, L):
    E = batch_energy_function(a_batch, D_batch, xi_batch, K_batch, L)
    return torch.exp(-E)




def batch_conditional_distribution(a_batch, D_batch, xi_batch, K_batch, L, sample_space):
    E = batch_energy_function(a_batch, D_batch, xi_batch, K_batch, L)
    Z = batch_partition_function(D_batch, xi_batch, K_batch, L, sample_space)
    return torch.exp(-E) / Z

def batch_partition_function(D_batch, xi_batch, K_batch, L, sample_space):
    Z = torch.zeros(D_batch.size(0), device=D_batch.device)
    for a in sample_space:
        Z += torch.exp(-batch_energy_function(a, D_batch, xi_batch, K_batch, L))
    return Z

def batch_calculate_weights(sample_space, D_batch, xi_batch, K_batch, L):
    weights = []
    for a_i in sample_space:
        weight = torch.exp(-batch_objective(a_i, D_batch, xi_batch, K_batch, L))
        weights.append(weight)
    weights = torch.stack(weights, dim=0)
    total_weight = torch.sum(weights, dim=0)
    return weights / total_weight

def batch_stage_two_weighted_loss(sample_space, D_batch, xi_batch, K_batch, L):
    weights = batch_calculate_weights(sample_space, D_batch, xi_batch, K_batch, L)
    loss = torch.zeros(D_batch.size(0), device=D_batch.device)
    for i, a in enumerate(sample_space):
        p = batch_conditional_distribution(a, D_batch, xi_batch, K_batch, L, sample_space)
        loss += weights[i] * torch.log(p)
    return -loss.mean()


# Rounded by a binary action matrix:
def rounded_by_binary_action(tensor, binary):
    rounded_ceil = torch.ceil(tensor)
    rounded_floor = torch.floor(tensor)
    result = torch.where(binary > 0.5, rounded_ceil, rounded_floor)
    return result


