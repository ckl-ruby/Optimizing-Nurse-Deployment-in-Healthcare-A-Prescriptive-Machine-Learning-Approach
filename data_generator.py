import coptpy as cp
from coptpy import COPT
import numpy as np
import argparse
import pandas
import pickle 
import os


def arg_prase():
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration.')
    parser.add_argument('--num_location', type=int, default=5, help='number of location in problem setup')
    parser.add_argument('--num_timeframe', type=int, default=20, help='number of timeframe in problem setup')
    parser.add_argument('--num_samples', type=int, default=51200, help='number of samples to generate')
    parser.add_argument('--saving_path', type=str, default="./data", help='dataset saving path')
    parser.add_argument('--loss_type', type=str, default="quad", help='loss type of g(a_t), options: [quad, excess]')
    return parser.parse_args()

def copt_solver(args):
    # Problem parameters
    L = args.num_location
    T = args.num_timeframe # Example number of terms, adjust as needed

    # Create COPT environment
    env = cp.Envr()

    # Create COPT model
    model = env.createModel("staff_transfer_optimization")

    # Add variables
    a_t = [[[model.addVar(lb=0.0, ub=COPT.INFINITY, name=f'a_{t}_{i}_{j}') for j in range(L)] for i in range(L)] for t in range(T)]

    # Create example demand and capacity data
    xi_t = np.random.rand(T, L)
    K = np.random.rand(L)

    # Add linear constraints
    for t in range(T):
        for i in range(L):
            model.addConstr(sum(a_t[t][i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{t}_{i}")
            for j in range(L):
                model.addConstr(a_t[t][i][j] >= 0, name=f"non_negativity_constraint_{t}_{i}_{j}")

    # Add quadratic objective
    D = np.random.rand(L, L)  # Example cost matrix, adjust as needed
    obj =cp.QuadExpr()  # Initialize obj as an expression

    # Add linear cost terms
    for t in range(T):
        for i in range(L):
            for j in range(L):
                obj += D[i, j] * a_t[t][i][j]

    # Add quadratic penalty terms
    for t in range(T):
        for i in range(L):
            delta_t_i = xi_t[t, i] - K[i] - sum(a_t[t][i][j] for j in range(L)) + sum(a_t[t][j][i] for j in range(L))
            if args.loss_type == "quad":
                obj += delta_t_i * delta_t_i
            elif args.loss_type == "excess":
                if delta_t_i >= 0:
                    obj += delta_t_i 

    model.setObjective(obj, COPT.MINIMIZE)

    # Set parameters
    model.setParam(COPT.Param.TimeLimit, 60)

    # Solve the problem
    model.solve()

    # Analyze solution
    if model.status == COPT.OPTIMAL:
        print("\nOptimal objective value: {0:.9e}".format(model.objval))
        vars = model.getVars()

        print("Variable solution:")
        solution_np = np.zeros((T,L,L))
        for var in vars:
            _, t, i, j = var.name.split('_')
            solution_np[int(t)][int(i)][int(j)] = var.x
            # print(" {0} = {1:.9e}".format(var.name, var.x))
        return L, T, xi_t, K, D, solution_np
    else:
        return None # Not optimal solution
    

def generate(args):
    dataset_X = {"L":[], "T":[], "xi_t":[], "K":[], "D":[]}
    dataset_y = {"solution":[]}
    for _ in range(args.num_samples):
        sample = copt_solver(args)
        if sample is not None:
            L, T, xi_t, K, D, solution = sample
            dataset_X["L"].append(L)
            dataset_X["T"].append(T)
            dataset_X["xi_t"].append(xi_t)
            dataset_X["K"].append(K)
            dataset_X["D"].append(D)
            dataset_y["solution"].append(solution)
    dataset_X["xi_t"] = np.array(dataset_X["xi_t"])
    dataset_y["solution"] = np.array(dataset_y["solution"])

    if not os.path.exists(args.saving_path): 
        os.makedirs(args.saving_path)

    with open(os.path.join(args.saving_path, f'X_{args.num_samples}_L{args.num_location}T{args.num_timeframe}_{args.loss_type}.pkl'), 'wb') as file:
        pickle.dump(dataset_X, file)

    with open(os.path.join(args.saving_path, f'y_{args.num_samples}_L{args.num_location}T{args.num_timeframe}_{args.loss_type}.pkl'), 'wb') as file:
        pickle.dump(dataset_y, file)

            
if __name__ == "__main__":
    args = arg_prase()
    generate(args)