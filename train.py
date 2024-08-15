import argparse
import pickle 
import os

from torch.utils.data import DataLoader, random_split
from  torch_geometric.loader.dataloader import DataLoader as GraphDataloader
import torch.optim as optim
import torch
import json
import matplotlib.pyplot as plt
import numpy as np
import joblib 

from scheduling_dataset import SchedulingDataset, SchedulingGraphDataset, SchedulingExntDataset, SchedulingGraphExntDataset
from model import SchedulingNN, TransformerRegressor, GNNModel, StageTwoNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import hybird_loss, exnt_loss
from torch.utils.data import Subset

from trainer import train, train_stage2_NN, inference_dtree, inference_NN, train_gnn, inference_GNN, inference_stage2_NN, train_stage2_GNN, inference_stage2_GNN

def arg_prase():
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration.')
    # data loading args
    parser.add_argument('--data_path', type=str, default="./data", help='dataset saving path')
    parser.add_argument('--data_filename', type=str, default="data_X_51200.pkl", help='dataset saving path')
    parser.add_argument('--labels_filename', type=str, default="data_y_51200.pkl", help='label saving path')
    parser.add_argument('--train_test_rate', type=float, default=0.8, help='the size rate of train to test set')
    parser.add_argument('--zero_thresholds', nargs='+',
                        default=["1e-4", "1e-3", "1e-2", "5e-2", "1e-1"], help='the list of thresholds of treating the number as zero on the solution.')
    parser.add_argument('--Kxi_bound', nargs='+', default=["10", "200"], help='bound of random input L and xi')
    parser.add_argument('--D_bound', nargs='+', default=["1", "2"], help='upper_bound of random input')

    parser.add_argument('--g_type', type=str, default="quadratic", help='label saving path')
    # Model selection
    parser.add_argument('--model_type', type=str, default="NN", help='model types: NN, stage2_NN, dtree')
    parser.add_argument('--first_model_type', type=str, default="GNN", help='model types: NN, stage2_NN, dtree')
    # stage2 args
    parser.add_argument('--stage1_model_path', type=str, default="./data", help='stage1_model_path, for the input of stage2')
    parser.add_argument('--round_threshold', type=float, default="0.5", help="the threshold of the ceilling action on making labels.")
    # traing args 
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint", help='checkpoint_path')
    parser.add_argument('--save_name_suffix', type=str, default="", help='add custom suffix on save name')
    parser.add_argument('--model_selection_strategy', nargs='+', 
                        default=["eval_loss", "f1", "infesibility", "cost_obj"], help='eval_loss, f1, infesibility')
    # loss settings
    parser.add_argument('--loss_combo', type=str,default="00010", 
                        help='choose which loss we use: 1.infesibility 2.diag_product 3.BCE 4.MSE')
    parser.add_argument('--loss_lambda', nargs='+',
                        default=["1", "1", "1", "1", "1"], 
                        help='the list of thresholds of treating the number as zero on the solution.')

    # optimizer
    parser.add_argument('--optimizer', type=str, default="adam", help='model types: NN, stage2_NN, dtree')
    # inference_step
    parser.add_argument('--inference', action='store_true', help="whether load from checkpoint or not")
    parser.add_argument('--inference_only', action='store_true', help="do inference_only")
    parser.add_argument('--inference_step', type=int, default=-1, help='inference_step')
    parser.add_argument('--inference_from_ckp', action='store_true', help="whether load from checkpoint or not")
    parser.add_argument('--inference_model_path', type=str, default="", help='inference_model path')
    # for NN
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--train_epoch', type=int, default=3, help='max training epoch')
    parser.add_argument('--train_step', type=int, default=5000, help='max training step')
    parser.add_argument('--train_bystep', action='store_true', help='train by step')
    parser.add_argument('--eval_step', type=int, default=50, help='evaluate between steps')
    parser.add_argument('--d_model', type=int, default=512, help='d_model')
    parser.add_argument('--mid_layer', type=int, default=5, help='middle layers')
    parser.add_argument('--hybird_loss', action='store_true', help="use hybird loss or not")
    parser.add_argument('--loss_scalar', type=float, default=1.1, help="scalar up the value that is not zero.")
    #Training args for Decision Tree / Random forest
    parser.add_argument('--max_depth', type=int, default=5, help='max_depth for decision')
    parser.add_argument('--n_estimator', type=int, default=50, help='number of estimators of random forest')

    # EXNT
    parser.add_argument('--exnt', action='store_true', help="train with ex ante dataset")
    parser.add_argument('--calculated_obj', action='store_true', help="use the calculated obj as ground truth or the solver obj")
    return parser.parse_args()

def plotting(save_path, train_log):
    steps = train_log["step"]
    train_loss = train_log["train_loss"]
    val_loss = train_log["eval_loss"]
    test_loss = train_log["test_loss"]
    plt.plot(steps, train_loss, linestyle='-', color="#F1B656", label='train_loss')
    plt.plot(steps, val_loss, linestyle='-', color="#397FC7", label='val_loss')
    plt.plot(steps, test_loss, linestyle='-', color="#040676", label='test_loss')

    plt.title("Loss by steps")
    plt.xlabel('Train Steps')
    plt.ylabel('Losses')
    plt.legend()

    plt.savefig(save_path)

def main():
    # cuda 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args = arg_prase()
    # load dataset
    print(os.path.join("data", args.data_filename))
    with open(os.path.join("data", args.data_filename), 'rb') as file:
        data = pickle.load(file)

    kxi_scale = float(args.Kxi_bound[1]) - float(args.Kxi_bound[0])
    d_scale = float(args.D_bound[1]) - float(args.D_bound[0])
    args.kxi_scale = kxi_scale
    args.d_scale = d_scale

    if args.model_type in ["GNN", "stage2_GNN"]:
        if args.exnt:
            dataset = SchedulingGraphExntDataset(data, kxi_scale, d_scale)
        else:
            dataset = SchedulingGraphDataset(data, kxi_scale, d_scale)
    else:
        if args.exnt:
            dataset = SchedulingExntDataset(data, kxi_scale, d_scale)
        else:
            dataset = SchedulingDataset(data, kxi_scale, d_scale)

    # train & test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    print(len(dataset))
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"train_size: {len(train_dataset)} \n val_size:{len(val_dataset)} \n test_size: {len(test_dataset)} \n")

    # data loader
    if args.model_type  in ['NN', "stage2_NN"]:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    elif args.model_type in ['GNN', "stage2_GNN"]:
        train_loader = GraphDataloader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = GraphDataloader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = GraphDataloader(test_dataset, batch_size=args.batch_size, shuffle=False)
    else:
        print("no such model! {args.model_type}")

    # load model
    if args.model_type == 'NN':
        input_dim, out_dim = dataset[0]["x"].shape[0], dataset[0]["y"].shape[0]
        model = SchedulingNN(input_dim, args.d_model, out_dim, args.mid_layer)
        model = model.to(device)
    elif args.model_type == 'transformer':
        input_dim, out_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]
        model = TransformerRegressor(input_dim=input_dim, d_model=args.d_model, 
                                     out_dim=out_dim, num_encoder_layers=args.mid_layer)
        model = model.to(device)
    elif args.model_type == "GNN":
        num_node, input_dim, out_dim = dataset[0][0].x.shape[0], dataset[0][0].x.shape[1], dataset[0][1].shape[0]
        model = GNNModel(in_channels=input_dim, d_model=args.d_model, 
                         out_channels=out_dim, num_node=num_node, mid_layer=args.mid_layer)
        model = model.to(device)
    elif args.model_type == "stage2_NN":
        input_dim, out_dim = dataset[0]["x"].shape[0], dataset[0]["y"].shape[0]
        stage1_model = SchedulingNN(input_dim, args.d_model, out_dim, args.mid_layer)
        stage1_model.load_state_dict(torch.load(args.stage1_model_path))
        stage1_model = stage1_model.to(device)

        model = StageTwoNN(input_dim + out_dim, args.d_model, out_dim, args.mid_layer) # inputs are stage1's input + a  
        model = model.to(device)
    elif args.model_type == "stage2_GNN":
        num_node, input_dim, out_dim, nn_feature_dim = dataset[0][0].x.shape[0], dataset[0][0].x.shape[1], dataset[0][1].shape[0], dataset[0][0].features_vector.shape[0]
        stage1_model = GNNModel(in_channels=input_dim, d_model=args.d_model, 
                         out_channels=out_dim, num_node=num_node, mid_layer=args.mid_layer)
        stage1_model.load_state_dict(torch.load(args.stage1_model_path))
        stage1_model = stage1_model.to(device)
        model = StageTwoNN(num_node**2 + nn_feature_dim, args.d_model, out_dim, args.mid_layer) # inputs are stage1's input + a  
        model = model.to(device)
    elif args.model_type == 'dtree':
        model = DecisionTreeRegressor(max_depth=args.max_depth)
    elif args.model_type == 'rdforest':
        model = RandomForestRegressor(n_estimators=args.n_estimator)
    else:
        print("No such model!")


    # train
    if args.model_type in ['NN', "transformer", "GNN", "stage2_NN", "stage2_GNN"]:
        # train task name
        training_task_name = f"{args.model_type}_on_{args.data_filename}_step{args.train_step}_bs{args.batch_size}_hd{args.d_model}_ly{args.mid_layer}_loss{args.loss_combo}_lambda{args.loss_lambda}_{args.optimizer}_lr{args.learning_rate}_round{args.round_threshold}{args.save_name_suffix}"
        # optimizer & loss
        if args.exnt:
            criterion = exnt_loss
        else:
            criterion = hybird_loss
        if args.optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == "adam":
            optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
        else:
            print("no such optimizer")
        train_args = {"train_loader": train_loader, 
                    "val_loader":val_loader, 
                    "test_loader": test_loader,
                    "model":model, 
                    "criterion":criterion, 
                    "optimizer":optimizer,
                    "args": args,
                    "task_name":training_task_name,
                    "device": device
        }


        # save training log as json
        task_save_path = os.path.join(args.checkpoint_path, training_task_name)
        if not args.inference_only:
            if args.model_type == 'NN': 
                train_log = train(**train_args)
            elif args.model_type == 'GNN':
                train_log = train_gnn(**train_args)
            elif args.model_type == "stage2_NN":
                train_args["stage1_model"] = stage1_model
                train_log = train_stage2_NN(**train_args)
            elif args.model_type == "stage2_GNN":
                train_args["stage1_model"] = stage1_model
                train_log = train_stage2_GNN(**train_args)
            else:
                raise f"no such model {args.model_type}"
    
       
            with open(os.path.join(task_save_path, "train_log.json"), "w+") as file:
                json.dump(train_log, file, indent=4)

            args_dict = vars(args)
            with open(os.path.join(task_save_path, "train_args.json"), "w+") as file:
                json.dump(args_dict, file, indent=4)

            plotting(os.path.join(task_save_path, "loss_plotting.pdf"), train_log)

            print("training completed!")

        # inference
        if args.inference:
            print("inferencing...")
            for selection_strategy in args.model_selection_strategy:
                print(f"loading the best model for {selection_strategy}")
                
                if args.inference_model_path != "":
                    task_save_path = args.inference_model_path
                ckp_path = os.path.join(task_save_path, f"best_{args.model_type}_{selection_strategy}.pth")
                if args.model_type == "NN":
                    inference_model = SchedulingNN(input_dim, args.d_model, out_dim, args.mid_layer)
                    inference_model.load_state_dict(torch.load(ckp_path))
                    inference_model.to(device)
                elif args.model_type == "stage2_NN":
                    inference_model = model = StageTwoNN(input_dim + out_dim, args.d_model, out_dim, args.mid_layer)
                    inference_model.load_state_dict(state_dict=torch.load(ckp_path))
                    inference_model.to(device)
                elif args.model_type == "stage2_GNN":
                    inference_model = model = StageTwoNN(num_node**2 + nn_feature_dim, args.d_model, out_dim, args.mid_layer)
                    inference_model.load_state_dict(state_dict=torch.load(ckp_path))
                    inference_model.to(device)
                elif args.model_type == "transformer":
                    inference_model = TransformerRegressor(input_dim=input_dim, d_model=args.d_model, 
                                                        out_dim=out_dim, num_encoder_layers=args.mid_layer)
                    inference_model.load_state_dict(torch.load(ckp_path))
                    inference_model.to(device)
                elif args.model_type == "GNN":
                    inference_model = GNNModel(in_channels=input_dim, d_model=args.d_model, out_channels=out_dim, num_node=num_node, mid_layer=args.mid_layer)
                    inference_model.load_state_dict(torch.load(ckp_path))
                    inference_model.to(device)
                else:
                    raise "no such model"
                
                inference_args = {
                    "model": inference_model, 
                    "tst_dataset": test_dataset, 
                    "device": device, 
                    "args": args, 
                    "inference_step": args.inference_step
                }
                for zero_threshold in args.zero_thresholds:
                    inference_args["zero_threshold"] = float(zero_threshold)
                    if args.model_type == "GNN":
                        inference = inference_GNN
                    elif args.model_type == "NN":
                        inference = inference_NN
                    elif args.model_type == "stage2_NN":
                        inference_args["stage1_model"] = stage1_model
                        inference = inference_stage2_NN
                    elif args.model_type == "stage2_GNN":
                        inference_args["stage1_model"] = stage1_model
                        inference = inference_stage2_GNN
                    
                    inference_log = inference(**inference_args)
                    with open(os.path.join(task_save_path, f"inference_result_{selection_strategy}_{zero_threshold}.json"), "w+") as file:
                        json.dump(inference_log, file, indent=4)
                    print(f"zero_threshold:{zero_threshold}: {inference_log}")
            print("inferencing completed!")

    elif args.model_type in ['dtree', 'rdforest']:
        training_task_name = f"{args.model_type}_on_{args.data_filename}_depth{args.max_depth}"
        features = []
        labels = []
        for features_vector, solution_vector, _, _ in train_dataset:
            features.append(features_vector.numpy())
            labels.append(solution_vector.numpy())
        train_dataset_X_np = np.array(features)
        train_dataset_Y_np = np.array(labels)
        print(train_dataset_X_np.shape, train_dataset_Y_np.shape)

        if not args.inference_only:
            print(f"fitting {model.__class__}......")
            model.fit(train_dataset_X_np, train_dataset_Y_np)
            print("saving model ......")

        task_save_path = os.path.join(args.checkpoint_path, training_task_name)
        if not os.path.exists(task_save_path):
            os.makedirs(task_save_path)
        joblib.dump(model, os.path.join(task_save_path, "dtree.joblib"))

        if args.inference:
            if args.inference_from_ckp:
                inference_model = joblib.load(os.path.join(task_save_path, "dtree.joblib"))
            else:
                inference_model = model
            predict_score, label_score = inference_dtree(inference_model, test_dataset, inference_step=args.inference_step)
            inference_log = {"predict_score": predict_score, "label_score": label_score}
            with open(os.path.join(task_save_path, "inference_result.json"), "w") as file:
                json.dump(inference_log, file, indent=4)
            print(inference_log)
    else:
        pass
    


if __name__ == "__main__":
    main()





