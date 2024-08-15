import coptpy as cp
from coptpy import COPT
import numpy as np
import argparse
import pickle 
import os
from tqdm import tqdm
import scipy.stats as stats
import gurobipy as gp
from gurobipy import GRB


def arg_prase():
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration.')
    parser.add_argument('--num_location', type=int, default=5, help='number of location in problem setup')
    parser.add_argument('--num_timeframe', type=int, default=20, help='number of timeframe in problem setup')
    parser.add_argument('--num_samples', type=int, default=51200, help='number of samples to generate')
    parser.add_argument('--saving_path', type=str, default="./data", help='dataset saving path')
    parser.add_argument('--loss_type', type=str, default="quadratic", help='loss type of g(a_t), options: [quadratic, excess]')
    parser.add_argument('--random_scalar', type=int, default=500, help='the scale of random input')
    parser.add_argument('--zero_threshold', type=float, default=1e-5, help='number of samples to generate')
    parser.add_argument('--keep_zero_solution', action='store_true', help="only generate solution with actions")
    parser.add_argument('--with_int_solution', action='store_true', help="with_int_solution")
    parser.add_argument('--Kxi_bound', nargs='+', default=["10", "200"], help='bound of random input L and xi')
    parser.add_argument('--D_bound', nargs='+', default=["1", "2"], help='upper_bound of random input')
    parser.add_argument('--with_quard_farc', action='store_true', help="only generate solution with actions")
    # EX ante parameters
    parser.add_argument('--exnt', action='store_true', help="generate ex ante data")
    parser.add_argument('--location_mu', nargs='+', default=["1", "2"], help='upper_bound of random input')
    parser.add_argument('--location_sigma', nargs='+', default=["1", "2"], help='upper_bound of random input')
    parser.add_argument('--exnt_sample_amount', type=int, default=1, help='exnt_sample_amount')
    return parser.parse_args()

def copt_solver_quadratic(args):
    # Problem parameters
    L = args.num_location
    # Define the range for uniform distribution
    kxi_lo, kxi_hi = float(args.Kxi_bound[0]), float(args.Kxi_bound[1])
    d_lo, d_hi = float(args.D_bound[0]), float(args.D_bound[1])
    # Create example demand and capacity data
    xi = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    K = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    D = np.random.rand(L, L) *(d_hi - d_lo) + d_lo
    if args.with_quard_farc:
        qfrac = np.random.rand(3) *(d_hi - d_lo) + d_lo
    else:
        qfrac = np.array([1, 0, 0])
    # Create COPT environment
    env = cp.Envr()
    # Create COPT model
    model = env.createModel("staff_transfer_optimization")
    # Add integer variables
    a = [[model.addVar(lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
    # Add linear constraints
    for i in range(L):
        model.addConstr(sum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")
        for j in range(L):
            model.addConstr(a[i][j] >= 0, name=f"non_negativity_constraint_{i}_{j}")
    # Add quadratic objective
    obj = cp.QuadExpr()
    # Add linear cost terms
    for i in range(L):
        for j in range(L):
            obj += D[i, j] * a[i][j]
    # Add quadratic penalty terms
    if args.with_quard_farc:
        for i in range(L):
            delta_i = xi[i] - K[i] + sum(a[i][j] - a[j][i] for j in range(L))
            obj += qfrac[0]* (delta_i * delta_i) + qfrac[1] *delta_i + qfrac[2]
    else:
        for i in range(L):
            delta_i = xi[i] - K[i] + sum(a[i][j] - a[j][i] for j in range(L))
            obj += delta_i * delta_i
    model.setObjective(obj, COPT.MINIMIZE)
    # Set parameters
    model.setParam(COPT.Param.TimeLimit, 60)
    # Solve the problem
    model.solve()
    float_solution = None
    if model.status == COPT.OPTIMAL:
        optimal_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                value = a[i][j].x
                optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
        if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
            float_solution = optimal_matrix

    if args.with_int_solution:
        # Create COPT environment
        env = cp.Envr()
        # Create COPT model
        model = env.createModel("staff_transfer_optimization")
        # Add integer variables
        a = [[model.addVar(vtype=COPT.INTEGER, lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
        # Add linear constraints
        for i in range(L):
            model.addConstr(sum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")
            for j in range(L):
                model.addConstr(a[i][j] >= 0, name=f"non_negativity_constraint_{i}_{j}")
        # Add quadratic objective
        obj = cp.QuadExpr()
        # Add linear cost terms
        for i in range(L):
            for j in range(L):
                obj += D[i, j] * a[i][j]
        # Add quadratic penalty terms
        for i in range(L):
            delta_i = xi[i] - K[i] + sum(a[i][j] - a[j][i] for j in range(L))
            obj += delta_i * delta_i
        model.setObjective(obj, COPT.MINIMIZE)
        # Set parameters
        model.setParam(COPT.Param.TimeLimit, 60)
        # Solve the problem
        model.solve()
        int_solution = None
        if model.status == COPT.OPTIMAL:
            optimal_matrix = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    value = a[i][j].x
                    optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
            if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
                int_solution = optimal_matrix
        if int_solution is not None and float_solution is not None:
            return L, xi, K, D, float_solution, int_solution, qfrac
        else:
            return None
    else:
        return (L, xi, K, D, float_solution, qfrac) if float_solution is not None else None
        
def copt_solver_excess(args):
    # Problem parameters
    L = args.num_location
    # Define the range for uniform distribution
    kxi_lo, kxi_hi = float(args.Kxi_bound[0]), float(args.Kxi_bound[1])
    d_lo, d_hi = float(args.D_bound[0]), float(args.D_bound[1])
    # Create example demand and capacity data
    xi = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    K = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    D = np.random.rand(L, L) *(d_hi - d_lo) + d_lo

    # Create COPT environment
    env = cp.Envr()
    # Create COPT model
    model = env.createModel("staff_transfer_optimization")
    # Add variables
    a = [[model.addVar(lb=0.0, ub=COPT.INFINITY, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
    #a = [[model.addVar(vtype=COPT.INTEGER, lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
    # Add linear constraints
    for i in range(L):
        model.addConstr(sum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")
        for j in range(L):
            model.addConstr(a[i][j] >= 0, name=f"non_negativity_constraint_{i}_{j}")
    # Add variables for delta_i and delta_positive
    delta = [model.addVar(lb=-COPT.INFINITY, ub=COPT.INFINITY, name=f'delta_{i}') for i in range(L)]
    delta_positive = [model.addVar(lb=0.0, ub=COPT.INFINITY, name=f'delta_positive_{i}') for i in range(L)]
    # Add constraints for delta_i based on its definition and delta_positive for ReLU
    for i in range(L):
        delta_i = xi[i] - K[i] + sum(a[i][j] for j in range(L)) - sum(a[j][i] for j in range(L))
        model.addConstr(delta[i] == delta_i, name=f"delta_definition_{i}")
        model.addConstr(delta_positive[i] >= delta[i], name=f"delta_positive_constraint_1_{i}")
        model.addConstr(delta_positive[i] >= 0, name=f"delta_positive_constraint_2_{i}")
    obj = cp.LinExpr()  # Initialize obj as an expression
    # Add linear cost terms
    for i in range(L):
        for j in range(L):
            obj += D[i, j] * a[i][j]
    # Add penalty terms using delta_positive
    for i in range(L):
        obj += delta_positive[i]
    model.setObjective(obj, COPT.MINIMIZE)
    # Set parameters
    model.setParam(COPT.Param.TimeLimit, 60)
    # Solve the problem
    model.solve()
    float_solution = None
    if model.status == COPT.OPTIMAL:
        optimal_matrix = np.zeros((L, L))
        for i in range(L):
            for j in range(L):
                value = a[i][j].x
                optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
        if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
            float_solution = optimal_matrix

    if args.with_int_solution:
         # Create COPT environment
        env = cp.Envr()
        # Create COPT model
        model = env.createModel("staff_transfer_optimization")
        # Add variables
        # a = [[model.addVar(lb=0.0, ub=COPT.INFINITY, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
        a = [[model.addVar(vtype=COPT.INTEGER, lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
        # Add linear constraints
        for i in range(L):
            model.addConstr(sum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")
            for j in range(L):
                model.addConstr(a[i][j] >= 0, name=f"non_negativity_constraint_{i}_{j}")
        # Add variables for delta_i and delta_positive
        delta = [model.addVar(lb=-COPT.INFINITY, ub=COPT.INFINITY, name=f'delta_{i}') for i in range(L)]
        delta_positive = [model.addVar(lb=0.0, ub=COPT.INFINITY, name=f'delta_positive_{i}') for i in range(L)]
        # Add constraints for delta_i based on its definition and delta_positive for ReLU
        for i in range(L):
            delta_i = xi[i] - K[i] + sum(a[i][j] for j in range(L)) - sum(a[j][i] for j in range(L))
            model.addConstr(delta[i] == delta_i, name=f"delta_definition_{i}")
            model.addConstr(delta_positive[i] >= delta[i], name=f"delta_positive_constraint_1_{i}")
            model.addConstr(delta_positive[i] >= 0, name=f"delta_positive_constraint_2_{i}")
        obj = cp.LinExpr()  # Initialize obj as an expression
        # Add linear cost terms
        for i in range(L):
            for j in range(L):
                obj += D[i, j] * a[i][j]
        # Add penalty terms using delta_positive
        for i in range(L):
            obj += delta_positive[i]
        model.setObjective(obj, COPT.MINIMIZE)
        # Set parameters
        model.setParam(COPT.Param.TimeLimit, 60)
        # Solve the problem
        model.solve()
        int_solution = None
        if model.status == COPT.OPTIMAL:
            optimal_matrix = np.zeros((L, L))
            for i in range(L):
                for j in range(L):
                    value = a[i][j].x
                    optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
            if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
                int_solution = optimal_matrix
        if int_solution is not None and float_solution is not None:
            return L, xi, K, D, float_solution, int_solution
        else:
            return None
    else:
        return (L, xi, K, D, float_solution) if float_solution is not None else None

def exnt_quadratic(args):
    L = args.num_location

    kxi_lo, kxi_hi = float(args.Kxi_bound[0]), float(args.Kxi_bound[1])
    d_lo, d_hi = float(args.D_bound[0]), float(args.D_bound[1])

    xi_mean = np.random.rand(L)*(kxi_hi - kxi_lo) + kxi_lo
    xi_variance = np.random.rand(L) *(d_hi - d_lo) + d_lo
    K = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    D = np.random.rand(L, L) *(d_hi - d_lo) + d_lo
    # D = np.zeros((L, L))
    # print(xi_mean, xi_variance, K, D)
    m = args.exnt_sample_amount  # Number of iterations (samples) for SAA

    model = gp.Model("staff_transfer_optimization")

    # Add variables
    a = [[model.addVar(lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
    #a = [[model.addVar(lb=0, vtype=GRB.INTEGER, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]

    # Add linear constraints
    for i in range(L):
        model.addConstr(gp.quicksum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")

    # Add quadratic objective
    obj = gp.QuadExpr()

    # Add linear cost terms
    for i in range(L):
        for j in range(L):
            obj += D[i, j] * a[i][j]

    # Add quadratic penalty terms using SAA
    for i in range(L):
        for _ in range(m):
            # Sample from the normal distribution with mean xi_mean[i] and variance xi_variance[i]
            mu, sigma = xi_mean[i], np.sqrt(xi_variance[i])
            lower, upper = 0, K[i]
            xi_sample = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
            # xi_sample = np.random.normal(xi_mean[i], np.sqrt(xi_variance[i]))
            # print(xi_sample, xi_sample_1)
            # Calculate the delta for this sample
            delta_i = xi_sample - K[i] + gp.quicksum(a[i][j] - a[j][i] for j in range(L))
            
            # Add quadratic term to the objective
            obj += (1/m)* delta_i * delta_i

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Set Gurobi parameters (e.g., time limit)
    model.Params.TimeLimit = 60

    # Optimize the model
    model.optimize()
    float_solution = None
    float_obj_value = None
    if model.status == GRB.OPTIMAL:
        optimal_matrix = np.zeros((L, L))
        float_obj_value = model.objVal
        for i in range(L):
            for j in range(L):
                value = a[i][j].x
                optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
        if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
            float_solution = optimal_matrix
    # print("????????????????????",float_solution)
    if args.with_int_solution:
        model = gp.Model("staff_transfer_optimization")

        # Add variables
        # a = [[model.addVar(lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
        a = [[model.addVar(lb=0, vtype=GRB.INTEGER, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]

        # Add linear constraints
        for i in range(L):
            model.addConstr(gp.quicksum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")

        # Add quadratic objective
        obj = gp.QuadExpr()

        # Add linear cost terms
        for i in range(L):
            for j in range(L):
                obj += D[i, j] * a[i][j]

        # Add quadratic penalty terms using SAA
        for i in range(L):
            for _ in range(m):
                # Sample from the normal distribution with mean xi_mean[i] and variance xi_variance[i]
                mu, sigma = xi_mean[i], np.sqrt(xi_variance[i])
                lower, upper = 0, K[i]
                xi_sample = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
                # xi_sample = np.random.normal(xi_mean[i], np.sqrt(xi_variance[i]))
                
                # Calculate the delta for this sample
                delta_i = xi_sample - K[i] + gp.quicksum(a[i][j] - a[j][i] for j in range(L))
                
                # Add quadratic term to the objective
                obj += (1/m) * delta_i * delta_i

        # Set the objective
        model.setObjective(obj, GRB.MINIMIZE)

        # Set Gurobi parameters (e.g., time limit)
        model.Params.TimeLimit = 60

        # Optimize the model
        model.optimize()
        int_solution = None
        int_obj_value = None
        if model.status == GRB.OPTIMAL:
            optimal_matrix = np.zeros((L, L))
            int_obj_value = model.objVal
            for i in range(L):
                for j in range(L):
                    value = a[i][j].x
                    optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
            if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
                int_solution = optimal_matrix
        if int_solution is not None and float_solution is not None:
            return L, xi_mean, xi_variance, K, D, float_solution, int_solution, float_obj_value, int_obj_value
        else:
            return None
    else:
        return (L, xi_mean, xi_variance, K, D, float_solution, float_solution, float_obj_value, float_obj_value) if float_solution is not None else None

def exnt_excess(args):
    L = args.num_location

    kxi_lo, kxi_hi = float(args.Kxi_bound[0]), float(args.Kxi_bound[1])
    d_lo, d_hi = float(args.D_bound[0]), float(args.D_bound[1])

    xi_mean = np.random.rand(L)*(kxi_hi - kxi_lo) + kxi_lo
    xi_variance = np.random.rand(L) *(d_hi - d_lo) + d_lo
    K = np.random.randint(kxi_lo, kxi_hi, (L,)).astype(float)
    D = np.random.rand(L, L) *(d_hi - d_lo) + d_lo
    # D = np.zeros((L, L))
    # print(xi_mean, xi_variance, K, D)
    m = args.exnt_sample_amount  # Number of iterations (samples) for SAA
    
    model = gp.Model("staff_transfer_optimization")

    # Add variables
    a = [[model.addVar(lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
    #a = [[model.addVar(lb=0, vtype=GRB.INTEGER, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]

    # Add linear constraints
    for i in range(L):
        model.addConstr(gp.quicksum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")

    # Add quadratic objective
    obj = gp.QuadExpr()

    # Add linear cost terms
    for i in range(L):
        for j in range(L):
            obj += D[i, j] * a[i][j]

    # Add ReLU penalty terms using SAA
    for i in range(L):
        for _ in range(m):
            # Sample from the normal distribution with mean xi_mean[i] and variance xi_variance[i]
            mu, sigma = xi_mean[i], np.sqrt(xi_variance[i])
            lower, upper = 0, K[i]
            xi_sample = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
            
            # Calculate the delta for this sample
            delta_i = xi_sample - K[i] + gp.quicksum(a[i][j] - a[j][i] for j in range(L))
            
            # Introduce a new variable to represent the ReLU of delta_i
            relu_delta_i = model.addVar(lb=0, name=f'relu_delta_{i}_{_}')
            
            # Add constraint: relu_delta_i >= delta_i
            model.addConstr(relu_delta_i >= delta_i, name=f"relu_constraint_1_{i}_{_}")
            
            # Add constraint: relu_delta_i >= 0
            model.addConstr(relu_delta_i >= 0, name=f"relu_constraint_2_{i}_{_}")
            
            # Add ReLU term to the objective
            obj += 0.01 * relu_delta_i

    # Set the objective
    model.setObjective(obj, GRB.MINIMIZE)

    # Set Gurobi parameters (e.g., time limit)
    model.Params.TimeLimit = 60

    # Optimize the model
    model.optimize()

    float_solution = None
    float_obj_value = None
    if model.status == GRB.OPTIMAL:
        optimal_matrix = np.zeros((L, L))
        float_obj_value = model.objVal
        for i in range(L):
            for j in range(L):
                value = a[i][j].x
                optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
        if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
            float_solution = optimal_matrix

    if args.with_int_solution:
        model = gp.Model("staff_transfer_optimization")

        # Add variables
        # a = [[model.addVar(lb=0, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]
        a = [[model.addVar(lb=0, vtype=GRB.INTEGER, name=f'a_{i}_{j}') for j in range(L)] for i in range(L)]

        # Add linear constraints
        for i in range(L):
            model.addConstr(gp.quicksum(a[i][j] for j in range(L)) <= K[i], name=f"capacity_constraint_{i}")

        # Add quadratic objective
        obj = gp.QuadExpr()

        # Add linear cost terms
        for i in range(L):
            for j in range(L):
                obj += D[i, j] * a[i][j]

        # Add ReLU penalty terms using SAA
        for i in range(L):
            for _ in range(m):
                # Sample from the normal distribution with mean xi_mean[i] and variance xi_variance[i]
                mu, sigma = xi_mean[i], np.sqrt(xi_variance[i])
                lower, upper = 0, K[i]
                xi_sample = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
                
                # Calculate the delta for this sample
                delta_i = xi_sample - K[i] + gp.quicksum(a[i][j] - a[j][i] for j in range(L))
                
                # Introduce a new variable to represent the ReLU of delta_i
                relu_delta_i = model.addVar(lb=0, name=f'relu_delta_{i}_{_}')
                
                # Add constraint: relu_delta_i >= delta_i
                model.addConstr(relu_delta_i >= delta_i, name=f"relu_constraint_1_{i}_{_}")
                
                # Add constraint: relu_delta_i >= 0
                model.addConstr(relu_delta_i >= 0, name=f"relu_constraint_2_{i}_{_}")
                
                # Add ReLU term to the objective
                obj += 0.01 * relu_delta_i

        # Set the objective
        model.setObjective(obj, GRB.MINIMIZE)

        # Set Gurobi parameters (e.g., time limit)
        model.Params.TimeLimit = 60

        # Optimize the model
        model.optimize()
        int_solution = None
        int_obj_value = None
        if model.status == GRB.OPTIMAL:
            optimal_matrix = np.zeros((L, L))
            int_obj_value = model.objVal
            for i in range(L):
                for j in range(L):
                    value = a[i][j].x
                    optimal_matrix[i, j] = value if value > args.zero_threshold else 0  
            if optimal_matrix.sum() > args.zero_threshold or args.keep_zero_solution: # if not zero solution, return
                int_solution = optimal_matrix
        if int_solution is not None and float_solution is not None:
            return L, xi_mean, xi_variance, K, D, float_solution, int_solution, float_obj_value, int_obj_value
        else:
            return None
    else:
        return (L, xi_mean, xi_variance, K, D, float_solution, float_solution, float_obj_value, float_obj_value) if float_solution is not None else None
    
def generate(args):
    dataset = {"L":[], "xi":[], "K":[], "D":[], "solution":[], "int_solution":[], "qfrac":[]}
    if args.exnt:
        dataset['mus']=[]
        dataset['sigmas']=[]
        dataset["float_obj"]=[]
        dataset["int_obj"]=[]
    train_bar = tqdm(total=args.num_samples)
    if args.loss_type == "quadratic":
        if args.exnt:
            solver = exnt_quadratic
        else:
            solver = copt_solver_quadratic
    elif args.loss_type == "excess":
        if args.exnt:
            solver = exnt_excess
        else:
            solver = copt_solver_excess
    else:
        raise RuntimeError("no such slover!")
    number_of_samples = 0
    while number_of_samples < args.num_samples:
        sample = solver(args)
        if sample is not None:
            if args.exnt:
                L, mus, sigmas, K, D, float_solution, int_solution, float_obj_value, int_obj_value = sample
                dataset["L"].append(L)
                dataset["mus"].append(mus)
                dataset["sigmas"].append(sigmas)
                dataset["K"].append(K)
                dataset["D"].append(D)
                dataset["solution"].append(float_solution)
                dataset["int_solution"].append(int_solution)
                dataset["float_obj"].append(float_obj_value)
                dataset["int_obj"].append(int_obj_value)
            else:
                L, xi, K, D, float_solution, int_solution, qfrac = sample
                dataset["L"].append(L)
                dataset["xi"].append(xi)
                dataset["K"].append(K)
                dataset["D"].append(D)
                dataset["solution"].append(float_solution)
                dataset["int_solution"].append(int_solution)
                dataset["qfrac"].append(qfrac)
            train_bar.update(1)
            number_of_samples += 1
    if not os.path.exists(args.saving_path): 
        os.makedirs(args.saving_path)
    if args.exnt:
        filename = f'EXNT_{args.num_samples}_L{args.num_location}_{args.loss_type}_D{args.D_bound}_zeros{args.keep_zero_solution}_intsolution{args.with_int_solution}_m{args.exnt_sample_amount}.pkl'
    else:
        filename = f'{args.num_samples}_L{args.num_location}_{args.loss_type}_kxi{args.Kxi_bound}_D{args.D_bound}_zeros{args.keep_zero_solution}_intsolution{args.with_int_solution}_qfrac{args.with_quard_farc}.pkl'
    with open(os.path.join(args.saving_path, filename), 'wb+') as file:
        pickle.dump(dataset, file)

            
if __name__ == "__main__":
    args = arg_prase()
    generate(args)

