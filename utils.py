import torch
import torch.nn as nn


def infeasibility(a_t, K):
    # input size: T L L
    T, L, _ = a_t.shape
    infesibility = 0
    for t in range(T):
        for i in range(L):
            # model.addConstr(sum(a_t[t][i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{t}_{i}")
            constrain_test =  sum(a_t[t][i][j] for j in range(L)) - K[i]
            if constrain_test > 0 :
                infesibility += constrain_test
            for j in range(L):
                # model.addConstr(a_t[t][i][j] >= 0, name=f"non_negativity_constraint_{t}_{i}_{j}")
                if a_t[t][i][j] < 0:
                    infesibility += abs(a_t[t][i][j])
    return infesibility

def infeasibility_loss(output, batch_labels, T, L, Ks):
    infesibility = torch.tensor(0.0).to(output.device)
    Ks = Ks.to(output.device)
    for t in range(T):
        for i in range(L):
            constrain_test =  torch.sum(output[ :, t, i, :], dim=-1) - Ks[:, i]
            infesibility += torch.sum(constrain_test[constrain_test > 0])
            for j in range(L):
                infesibility += torch.sum(torch.abs(output[:, t, i, j][output[:, t, i, j] < 0]))
    diag_product_sum = torch.sum(output * output.transpose(3,2)) / sum(output.shape)
    output = output.reshape(-1, T *L * L)
    loss1 = 0.5 * (infesibility / sum(output.shape))
    loss2 = 1.0 * (nn.MSELoss()(output, batch_labels))
    loss3 = 0.5 * (diag_product_sum)
    return loss1 + loss2 + loss3
    
