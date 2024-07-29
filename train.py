import argparse
import pandas
import pickle 
import os
import matplotlib
from tqdm import tqdm

from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn as nn
import torch.optim as optim
import torch
import copy, json
import matplotlib.pyplot as plt
import numpy as np
import joblib 

from data_loader import SchedulingDataset
from model import SchedulingNN
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from utils import infeasibility, infeasibility_loss

def arg_prase():
    parser = argparse.ArgumentParser(description='Neural Network Training Configuration.')
    # data loading args
    parser.add_argument('--data_path', type=str, default="./data", help='dataset saving path')
    parser.add_argument('--data_filename', type=str, default="data_X_51200.pkl", help='dataset saving path')
    parser.add_argument('--labels_filename', type=str, default="data_y_51200.pkl", help='label saving path')
    parser.add_argument('--train_test_rate', type=float, default=0.8, help='the size rate of train to test set')

    # Model selection
    parser.add_argument('--model_type', type=str, default="NN", help='model types: NN, dtree')
    # traing args 
    parser.add_argument('--checkpoint_path', type=str, default="./checkpoint", help='checkpoint_path')
    # inference_step
    parser.add_argument('--inference', action='store_true', help="whether load from checkpoint or not")
    parser.add_argument('--inference_only', action='store_true', help="do inference_only")
    parser.add_argument('--inference_step', type=int, default=-1, help='inference_step')
    parser.add_argument('--inference_from_ckp', action='store_true', help="whether load from checkpoint or not")
    # for NN
    parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning_rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--train_epoch', type=int, default=3, help='max training epoch')
    parser.add_argument('--train_step', type=int, default=5000, help='max training step')
    parser.add_argument('--train_bystep', action='store_true', help='train by step')
    parser.add_argument('--eval_step', type=int, default=50, help='evaluate between steps')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    parser.add_argument('--infeasibility_loss', action='store_true', help="use infeasibility loss or not")
    #Training args for Decision Tree / Random forest
    parser.add_argument('--max_depth', type=int, default=5, help='max_depth for decision')
    parser.add_argument('--n_estimator', type=int, default=50, help='number of estimators of random forest')
    return parser.parse_args()

def save_args(args, save_path):
    args_dict = vars(args)
    if not os.path.exists(os.path.join(save_path, 'training_args.json')):
        os.makedirs(os.path.join(save_path, 'training_args.json'))
    with open(os.path.join(save_path, 'training_args.json'), 'w') as json_file:
        json.dump(args_dict, json_file, indent=4)

def train(train_loader, val_loader, test_loader, model, criterion, optimizer, args, task_name, device):
    global_step = 0
    train_log = {"epoch":[], "step":[], "train_loss":[], "eval_loss":[], "test_loss":[],
                 "min_eval_loss": float('inf'), "best_step":0}
    train_bar = tqdm(total=args.train_step)
    while global_step <= args.train_step:
        model.train()
        for batch_data, batch_labels, features, solution in train_loader:
            batch_data, batch_labels = batch_data.float().to(device), batch_labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(batch_data)
            if args.infeasibility_loss:
                T = features["T"][0]
                L = features["L"][0]        
                Ks = features["K"]
                outputs = outputs.reshape((-1, T, L, L))
                loss = criterion(outputs, batch_labels, T, L, Ks)
            else:
                loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_bar.update(1)
            train_bar.set_postfix(loss = loss.item())

            if global_step % args.eval_step == 0 or global_step == 1: # eval model when reach certain step
                val_loss = test(val_loader, model, criterion, device, args)
                test_loss = test(test_loader, model, criterion, device, args)
                train_log["epoch"].append(global_step / len(train_loader))
                train_log["step"].append(global_step)
                train_log["train_loss"].append(loss.item())
                train_log["eval_loss"].append(val_loss)
                train_log["test_loss"].append(test_loss)
                if val_loss < train_log["min_eval_loss"]: # if the eval loss is minimum, save the model as the best model.
                    train_log["min_eval_loss"] = val_loss
                    train_log["best_step"] = global_step
                    task_save_path = os.path.join(args.checkpoint_path, task_name)
                    if not os.path.exists(task_save_path):
                        os.makedirs(task_save_path)
                    torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_model.pth"))

            if global_step >= args.train_step: 
                break
        train_bar.close
    return train_log
    
def test(tst_loader, model, criterion, device, args):
    model.eval()
    tst_loss = 0.0
    eval_loop = tqdm((tst_loader), total = len(tst_loader))
    with torch.no_grad():
        for batch_data, batch_labels, features, solution in eval_loop:
            batch_data, batch_labels = batch_data.float().to(device), batch_labels.float().to(device)
            outputs = model(batch_data)
            if args.infeasibility_loss:
                T = features["T"][0]
                L = features["L"][0]        
                Ks = features["K"]
                outputs = outputs.reshape((-1, T, L, L))
                loss = criterion(outputs, batch_labels, T, L, Ks)
            else:
                loss = criterion(outputs, batch_labels)
            tst_loss += loss.item()
            eval_loop.set_postfix(loss=loss.item())
    tst_loss /= len(tst_loader)
    eval_loop.close
    return tst_loss

def inference_NN(model, tst_dataset, device, inference_step=-1):
    model.eval()
    loop = tqdm((tst_dataset), total = len(tst_dataset))
    predict_score = 0
    label_score = 0
    inference_count = 0
    with torch.no_grad():
        for idx, (features_vector, _, features, solution) in enumerate(loop):
            # model output
            features_vector= features_vector.float().to(device)
            outputs = model(features_vector)
            outputs = outputs.reshape(solution.shape)

            predict_score += infeasibility(outputs.cpu().numpy(), features["K"])
            label_score += infeasibility(solution, features["K"])
            inference_count += 1
            if inference_step != -1 and idx >= inference_step:
                break
    
    predict_score /= inference_count
    label_score /= inference_count
    return predict_score, label_score

def inference_dtree(model, tst_dataset, inference_step=-1):
    tst_dataset_X = []
    y_labels = []
    features_list = []
    inference_count = 0
    predict_score = 0
    label_score = 0
    for idx, (features_vector, _, features, solution) in enumerate(tst_dataset):
        tst_dataset_X.append(features_vector.numpy())
        y_labels.append(solution)
        features_list.append(features)
        inference_count += 1
        if inference_step != -1 and idx >= inference_step:
            break
    train_dataset_X_np = np.array(tst_dataset_X)

    y_predicted = model.predict(train_dataset_X_np)
    for idx, (y_p, y_l) in enumerate(zip(y_predicted, y_labels)):
        y_p = y_p.reshape(y_l.shape)
        predict_score += infeasibility(y_p, features_list[idx]["K"])
        label_score += infeasibility(y_l, features_list[idx]["K"])

    predict_score /= inference_count
    label_score /= inference_count
    return predict_score, label_score

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
    with open(os.path.join("data", args.data_filename), 'rb') as file:
        data_X = pickle.load(file)

    with open(os.path.join("data", args.labels_filename), 'rb') as file:
        data_y = pickle.load(file)

    dataset = SchedulingDataset(data_X, data_y)

    # train & test split
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    print(f"train_size: {len(train_dataset)} \n val_size:{len(val_dataset)} \n test_size: {len(test_dataset)} \n")

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # load model
    if args.model_type == 'NN':
        input_dim, out_dim = train_dataset[0][0].shape[0], train_dataset[0][1].shape[0]
        model = SchedulingNN(input_dim, args.hidden_dim, out_dim)
        model = model.to(device)
    elif args.model_type == 'dtree':
        model = DecisionTreeRegressor(max_depth=args.max_depth)
    elif args.model_type == 'rdforest':
        model =RandomForestRegressor(n_estimators=args.n_estimator)
    else:
        print("No such model!")


    # train
    if args.model_type == 'NN':
        # train task name
        training_task_name = f"{args.model_type}_on_{args.data_filename}_step{args.train_step}_bs{args.batch_size}"
        # optimizer & loss
        if args.infeasibility_loss:
            criterion = infeasibility_loss
        else:
            criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)

        train_args = {"train_loader": train_loader, 
                    "val_loader":val_loader, 
                    "test_loader":test_loader,
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
            train_log = train(**train_args)
    
       
            with open(os.path.join(task_save_path, "train_log.json"), "w") as file:
                json.dump(train_log, file, indent=4)

            # with open(os.path.join(task_save_path, "train_log.json"), "r") as file:
            #     train_log = json.load(file)
            plotting(os.path.join(task_save_path, "loss_plotting.pdf"), train_log)

        # inference
        if args.inference:
            if args.inference_from_ckp:
                inference_model = SchedulingNN(input_dim, args.hidden_dim, out_dim).load_state_dict(torch.load(args.ckp_path))
            else:
                inference_model = model
            predict_score, label_score = inference_NN(inference_model, test_dataset, device, inference_step=args.inference_step)
            inference_log = {"predict_score": predict_score, "label_score": label_score}
            with open(os.path.join(task_save_path, "inference_result.json"), "w") as file:
                json.dump(inference_log, file, indent=4)
            print(inference_log)
        save_args(args, training_task_name)

    
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
        save_args(args, training_task_name)
    else:
        pass
    


if __name__ == "__main__":
    main()





