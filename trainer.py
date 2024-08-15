
import os
from tqdm import tqdm

import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import f1_score, confusion_matrix
from utils import infeasibility, minimize_objective_quardratic, minimize_objective_excess, rounded_by_binary_action, zero_refine, zero_refine_batch, minimize_objective_quardratic_exnt

from  torch_geometric.loader.dataloader import DataLoader as GraphDataloader


def train(train_loader, val_loader, test_loader, model, criterion, optimizer, args, task_name, device):
    global_step = 0
    train_log = {"epoch":[], "step":[], "train_loss":[], "eval_loss":[], "test_loss":[],
                 "min_eval_loss": float('inf'), 
                 "min_cost_objective": float('inf'),
                 "max_f1_score": float(0),
                 "min_infesibility": float('inf'),
                "inference_log":[]}
    train_bar = tqdm(total=args.train_step)
    while global_step <= args.train_step:
        model.train()
        for data_batch in train_loader:
            if args.exnt:
                x, y, K, mu, sigma, D, L, int_y, y_float_obj, y_int_obj = (data_batch["x"].to(device), 
                       data_batch["y"].to(device), 
                       data_batch["x_origin"]["K"].to(device),
                       data_batch["x_origin"]["mus"].to(device),
                       data_batch["x_origin"]["sigmas"].to(device),
                       data_batch["x_origin"]["D"].to(device),
                       data_batch["x_origin"]["L"],
                       data_batch["int_y"].to(device),
                       data_batch["x_origin"]["float_obj"].to(device),
                       data_batch["x_origin"]["int_obj"].to(device))
                optimizer.zero_grad()
                outputs = model(x)
                if args.calculated_obj:
                    loss = criterion(outputs, y, K, mu, sigma, D, L, args)
                else:
                    loss = criterion(outputs, y_float_obj, K, mu, sigma, D, L, args)
            else:
                x, y, K, xi, D, L, qfrac= (data_batch["x"].to(device), 
                        data_batch["y"].to(device), 
                        data_batch["x_origin"]["K"].to(device),
                        data_batch["x_origin"]["xi"].to(device),
                        data_batch["x_origin"]["D"].to(device),
                        data_batch["x_origin"]["L"],
                        data_batch["x_origin"]["qfrac"])
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y, K, xi, D, L, args, qfrac=qfrac)
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
                inference_log = inference_NN(model, val_loader.dataset, device, 
                                                zero_threshold=float("1e-2"), args=args, inference_step=args.inference_step)
                train_log["inference_log"].append(inference_log)
                if "eval_loss" in args.model_selection_strategy:
                    if val_loss < train_log["min_eval_loss"]: # if the eval loss is minimum, save the model as the best model.
                        train_log["min_eval_loss"] = val_loss
                        train_log["best_step_eval"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_eval_loss.pth"))
                if "f1" in args.model_selection_strategy:
                    if inference_log["f1_score"] > train_log["max_f1_score"]: # if the f1_score is max, save the model as the best model.
                        train_log["max_f1_score"] = inference_log["f1_score"]
                        train_log["best_step_f1"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_f1.pth"))
                if "infesibility" in args.model_selection_strategy:
                    if inference_log["predict_score"] < train_log["min_infesibility"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_infesibility"] = inference_log["predict_score"]
                        train_log["best_step_infesibility"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_infesibility.pth")) 
                if "cost_obj" in args.model_selection_strategy:
                    if inference_log["minimize_objective_predict"] < train_log["min_cost_objective"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_cost_objective"] = inference_log["minimize_objective_predict"]
                        train_log["best_step_cost_obj"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_cost_obj.pth"))          
            if global_step >= args.train_step: 
                break
        train_bar.close
    return train_log

def test(tst_loader, model, criterion, device, args):
    model.eval()
    tst_loss = 0.0
    eval_loop = tqdm((tst_loader), total = len(tst_loader))
    with torch.no_grad():
        for data_batch in eval_loop:
            if args.exnt:
                x, y, K, mu, sigma, D, L, int_y, y_float_obj, y_int_obj = (data_batch["x"].to(device), 
                       data_batch["y"].to(device), 
                       data_batch["x_origin"]["K"].to(device),
                       data_batch["x_origin"]["mus"].to(device),
                       data_batch["x_origin"]["sigmas"].to(device),
                       data_batch["x_origin"]["D"].to(device),
                       data_batch["x_origin"]["L"],
                       data_batch["int_y"].to(device),
                       data_batch["x_origin"]["float_obj"].to(device),
                       data_batch["x_origin"]["int_obj"].to(device))
                outputs = model(x)
                if args.calculated_obj:
                    loss = criterion(outputs, y, K, mu, sigma, D, L)
                else:
                    loss = criterion(outputs, y_float_obj, K, mu, sigma, D, L, args)
            else:
                x, y, K, xi, D, L, qfrac= (data_batch["x"].to(device), 
                        data_batch["y"].to(device), 
                        data_batch["x_origin"]["K"].to(device),
                        data_batch["x_origin"]["xi"].to(device),
                        data_batch["x_origin"]["D"].to(device),
                        data_batch["x_origin"]["L"],
                        data_batch["x_origin"]["qfrac"])
                outputs = model(x)
                loss = criterion(outputs, y, K, xi, D, L, args, qfrac=qfrac)
            tst_loss += loss.item()
            eval_loop.set_postfix(loss=loss.item())
    tst_loss /= len(tst_loader)
    eval_loop.close
    return tst_loss

def inference_NN(model, tst_dataset, device, args, zero_threshold, inference_step=-1):
    model.eval()
    loop = tqdm((tst_dataset), total = len(tst_dataset))
    predict_score = 0
    predict_score_zero_pair = 0
    predict_non_zero_count = 0
    label_score = 0
    label_non_zero_count = 0
    inference_count = 0
    mse_loss = 0
    f1, tn, fp, fn, tp = 0, 0, 0, 0, 0
    f1_zero_refine, tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = 0, 0, 0, 0, 0
    minimize_objective_predict = 0
    minimize_objective_predict_zero_pair = 0
    minimize_objective__label = 0
    with torch.no_grad():
        for idx, data in enumerate(loop):
            # model output
            if args.exnt:
                x, y, K, mu, sigma, D, L = (data["x"].to(device), 
                       data["y"].to(device), 
                       torch.tensor(data["x_origin"]["K"]).to(device),
                       torch.tensor(data["x_origin"]["mus"]).to(device),
                       torch.tensor(data["x_origin"]["sigmas"]).to(device),
                       torch.tensor(data["x_origin"]["D"]).to(device),
                       data["x_origin"]["L"])

                outputs = model(x.unsqueeze(0)).reshape(L, L)
            else:
                x, y, K, xi, D, L, qfrac = (data["x"].to(device), 
                        data["y"].to(device), 
                        torch.tensor(data["x_origin"]["K"]).to(device),
                        torch.tensor(data["x_origin"]["xi"]).to(device),
                        torch.tensor(data["x_origin"]["D"]).to(device),
                        data["x_origin"]["L"],
                        torch.tensor(data["x_origin"]["qfrac"]).to(device))
                outputs = model(x.unsqueeze(0)).reshape(L, L)
            y = y.reshape(L, L)
            predict_score += infeasibility(outputs, K)
            predict_score_zero_pair += infeasibility(zero_refine(outputs), K)
            predict_non_zero_count += torch.sum(outputs > zero_threshold).item()
            label_score += infeasibility(y, K)
            label_non_zero_count += torch.sum(y > zero_threshold).item()
            mse_loss += nn.MSELoss()(outputs, y).item()

            # f1_score
            solution_binary = torch.where(y < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_binary = torch.where(outputs < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_zero_refine_binary = torch.where(zero_refine(outputs).flatten().unsqueeze(0) < zero_threshold, torch.tensor(0), torch.tensor(1))

            f1 += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_binary.flatten().cpu().numpy())
            conf_matrix = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn, fp, fn, tp = tuple( x + y for x , y in zip((tn, fp, fn, tp), conf_matrix))

            f1_zero_refine += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1])
            conf_matrix_zero_refine = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = tuple( x + y for x , y in zip((tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine), conf_matrix_zero_refine))

            inference_count += 1
            if inference_step != -1 and idx >= inference_step:
                break

            # minimize_objective_
            # xi, K, D, outputs, y = xi, K, D, outputs*args.kxi_scale, y*args.kxi_scale
            # print(xi, K, D, outputs, y)
            # outputs, y = outputs*data["y_norm"], y*data["y_norm"]
            if args.g_type == "quadratic":
                if args.exnt:
                    obj_predict = minimize_objective_quardratic_exnt(D, outputs.reshape(L, L),  mu, sigma, K)
                    obj_predict_zero_pair = minimize_objective_quardratic_exnt(D, zero_refine(outputs).reshape(L, L),  mu, sigma, K)
                    obj_label = minimize_objective_quardratic_exnt(D, y.reshape(L, L),  mu, sigma, K)
                else:
                    fqurd = torch.tensor(data["x_origin"]["qfrac"]).to(device)
                    obj_predict = minimize_objective_quardratic(D, outputs.reshape(L, L), xi, K, fqurd)
                    obj_predict_zero_pair = minimize_objective_quardratic(D, zero_refine(zero_refine(outputs).reshape(L, L)), xi, K, fqurd)
                    obj_label = minimize_objective_quardratic(D, y.reshape(L, L), xi, K, fqurd)
            elif args.g_type == "excess":
                obj_predict = minimize_objective_excess(D, outputs.reshape(L, L), xi, K)
                obj_predict_zero_pair = minimize_objective_excess(D, zero_refine(zero_refine(outputs).reshape(L, L)), xi, K)
                obj_label = minimize_objective_excess(D, y.reshape(L, L), xi, K)
            else:
                print("no such d_type {args.g_type}")
            
            minimize_objective_predict += obj_predict
            minimize_objective_predict_zero_pair += obj_predict_zero_pair
            minimize_objective__label += obj_label

    predict_score /= inference_count
    predict_score_zero_pair /= inference_count
    predict_non_zero_count /= inference_count
    label_score /= inference_count
    label_non_zero_count /= inference_count
    mse_loss /= inference_count
    f1 /= inference_count
    tn /= inference_count
    fp /= inference_count
    fn /= inference_count
    tp /= inference_count
    f1_zero_refine /= inference_count
    tn_zero_refine /= inference_count
    fp_zero_refine /= inference_count
    fn_zero_refine /= inference_count
    tp_zero_refine /= inference_count
    minimize_objective_predict /= inference_count
    minimize_objective_predict_zero_pair /= inference_count
    minimize_objective__label /= inference_count
    return {
            "g_type": args.g_type,
            "inference_step": args.inference_step,
            "predict_score": predict_score.item(),
            "predict_score_zero_pair": predict_score_zero_pair.item(),
            "predict_non_zero_count": predict_non_zero_count,
            "label_score": label_score.item(),
            "label_non_zero_count": label_non_zero_count,
            "mse_loss": mse_loss,
            "zero_threshold": zero_threshold,
            "f1_score": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "f1_zero_refine": f1_zero_refine, 
            "tn_zero_refine": tn_zero_refine, 
            "fp_zero_refine": fp_zero_refine, 
            "fn_zero_refine": fn_zero_refine, 
            "tp_zero_refine": tp_zero_refine,
            "total_value": y.numel(),
            "minimize_objective_predict": minimize_objective_predict.item(),
            "minimize_objective_predict_zero_pair": minimize_objective_predict_zero_pair.item(),
            "minimize_objective__label": minimize_objective__label.item()
            }


def train_gnn(train_loader, val_loader, test_loader, model, criterion, optimizer, args, task_name, device):
    global_step = 0
    train_log = {"epoch":[], "step":[], "train_loss":[], "eval_loss":[], "test_loss":[],
                 "min_eval_loss": float('inf'), 
                 "min_cost_objective": float('inf'),
                 "max_f1_score": float(0),
                 "min_infesibility": float('inf'),
                "inference_log":[]}
    train_bar = tqdm(total=args.train_step)
    while global_step <= args.train_step:
        model.train()
        for databatch in train_loader: 
            data, y = databatch
            if args.exnt:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, mu, sigma, D, int_y, y_float_obj, y_int_obj = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["mus"])).to(device),
                    torch.tensor(np.array(data.features["sigmas"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.int_y)).to(device),
                    torch.tensor(np.array(data.features["float_obj"])).to(device),
                    torch.tensor(np.array(data.features["int_obj"])).to(device),
                )

                optimizer.zero_grad()

                outputs = model(x, edge_index, edge_attr, data.batch.to(device))
                if args.calculated_obj:
                    loss = criterion(outputs, y, K, mu, sigma, D, L, args)
                else:
                    loss = criterion(outputs, y_float_obj, K, mu, sigma, D, L, args)
            else:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, xi, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["qfrac"])).to(device)
                )

                optimizer.zero_grad()

                outputs = model(x, edge_index, edge_attr, data.batch.to(device))
                loss = criterion(outputs, y, K, xi, D, L, args, qfrac=qfrac)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_bar.update(1)
            train_bar.set_postfix(loss = loss.item())
            # ------------------------------

            if global_step % args.eval_step == 0 or global_step == 1: # eval model when reach certain step
                val_loss = test_gnn(val_loader, model, criterion, device, args)
                test_loss = test_gnn(test_loader, model, criterion, device, args)
                train_log["epoch"].append(global_step / len(train_loader))
                train_log["step"].append(global_step)
                train_log["train_loss"].append(loss.item())
                train_log["eval_loss"].append(val_loss)
                train_log["test_loss"].append(test_loss)
                inference_log = inference_GNN(model, val_loader.dataset, device, 
                                                zero_threshold=float("1e-2"), args=args, inference_step=args.inference_step)
                train_log["inference_log"].append(inference_log)
                if "eval_loss" in args.model_selection_strategy:
                    if val_loss < train_log["min_eval_loss"]: # if the eval loss is minimum, save the model as the best model.
                        train_log["min_eval_loss"] = val_loss
                        train_log["best_step_eval"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_eval_loss.pth"))
                if "f1" in args.model_selection_strategy:
                    if inference_log["f1_score"] > train_log["max_f1_score"]: # if the f1_score is max, save the model as the best model.
                        train_log["max_f1_score"] = inference_log["f1_score"]
                        train_log["best_step_f1"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_f1.pth"))
                if "infesibility" in args.model_selection_strategy:
                    if inference_log["predict_score"] < train_log["min_infesibility"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_infesibility"] = inference_log["predict_score"]
                        train_log["best_step_infesibility"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_infesibility.pth")) 
                if "cost_obj" in args.model_selection_strategy:
                    if inference_log["minimize_objective_predict"] < train_log["min_cost_objective"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_cost_objective"] = inference_log["minimize_objective_predict"]
                        train_log["best_step_cost_obj"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_cost_obj.pth"))          
            if global_step >= args.train_step: 
                break
        train_bar.close
    return train_log

def test_gnn(tst_loader, model, criterion, device, args):
    model.eval()
    tst_loss = 0.0
    train_bar = tqdm((tst_loader), total = len(tst_loader))
    with torch.no_grad():
        for databatch in train_bar: 
            data, y = databatch
            if args.exnt:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, xi, D, mu, sigma, y_float_obj = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["mu"])).to(device),
                    torch.tensor(np.array(data.features["sigma"])).to(device),
                    torch.tensor(np.array(data.features["float_obj"])).to(device),
                )


                outputs = model(x, edge_index, edge_attr, data.batch.to(device))
                if args.calculated_obj:
                    loss = criterion(outputs, y, K, mu, sigma, D, L, args)
                else:
                    loss = criterion(outputs, y_float_obj, K, mu, sigma, D, L, args)
            else:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, xi, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["qfrac"])).to(device),
                )

                outputs = model(x, edge_index, edge_attr, data.batch.to(device))
                loss = criterion(outputs, y, K, xi, D, L, args, qfrac = qfrac)
            train_bar.set_postfix(loss = loss.item())
    tst_loss /= len(tst_loader)
    train_bar.close
    return tst_loss

def inference_GNN(model, tst_dataset, device, args, zero_threshold, inference_step=-1):
    model.eval()
    tst_loader =  GraphDataloader(tst_dataset, batch_size=1, shuffle=False)
    loop = tqdm((tst_loader), total = len(tst_loader))
    predict_score = 0
    predict_score_zero_pair = 0
    predict_non_zero_count = 0
    label_score = 0
    label_non_zero_count = 0
    inference_count = 0
    mse_loss = 0
    f1, tn, fp, fn, tp = 0, 0, 0, 0, 0
    f1_zero_refine, tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = 0, 0, 0, 0, 0
    minimize_objective_predict = 0
    minimize_objective_predict_zero_pair = 0
    minimize_objective__label = 0
    with torch.no_grad():
        for idx, databatch in enumerate(loop):
            # model output
            data, y = databatch
            if args.exnt:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, xi, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["qfrac"]).to(device))
                )


                y = y.reshape(L, L)
                outputs = model(x, edge_index, edge_attr, data.batch.to(device)).reshape(L, L)
            else:
                x, edge_index, edge_attr, y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device) 
                )
                L, K, xi, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["qfrac"])).to(device)
                )
                y = y.reshape(L, L)
                outputs = model(x, edge_index, edge_attr, data.batch.to(device)).reshape(L, L)

            predict_score += infeasibility(outputs, K)
            predict_non_zero_count += torch.sum(outputs > zero_threshold).item()
            label_score += infeasibility(y, K)
            label_non_zero_count += torch.sum(y > zero_threshold).item()
            mse_loss += nn.MSELoss()(outputs, y).item()

            # f1_score
            solution_binary = torch.where(y < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_binary = torch.where(outputs < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_zero_refine_binary = torch.where(zero_refine(outputs).flatten().unsqueeze(0) < zero_threshold, torch.tensor(0), torch.tensor(1))

            f1 += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_binary.flatten().cpu().numpy(), zero_division=0)
            conf_matrix = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()

            tn, fp, fn, tp = tuple( x + y for x , y in zip((tn, fp, fn, tp), conf_matrix))

            f1_zero_refine += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_zero_refine_binary.flatten().cpu().numpy(), zero_division=0)
            conf_matrix_zero_refine = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = tuple( x + y for x , y in zip((tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine), conf_matrix_zero_refine))

            inference_count += 1
            if inference_step != -1 and idx >= inference_step:
                break

            # minimize_objective_
            if args.g_type == "quadratic":
                if args.exnt:
                    D, outputs, mu, sigma, K = D.reshape(L, L), outputs.reshape(L, L), mu.flatten(), sigma.flatten(), K.flatten()
                    obj_predict = minimize_objective_quardratic_exnt(D, outputs.reshape(L, L),  mu, sigma, K)
                    obj_predict_zero_pair = minimize_objective_quardratic_exnt(D, zero_refine(outputs).reshape(L, L),  mu, sigma, K)
                    obj_label = minimize_objective_quardratic_exnt(D, y.reshape(L, L),  mu, sigma, K)
                else:
                    qfrac = torch.tensor(np.array(data.features["qfrac"])).to(device)
                    D, outputs, xi, K = D.reshape(L, L), outputs.reshape(L, L), xi.flatten(), K.flatten()
                    obj_predict = minimize_objective_quardratic(D, outputs.reshape(L, L), xi, K, qfrac[0])
                    obj_predict_zero_pair = minimize_objective_quardratic(D, zero_refine(zero_refine(outputs).reshape(L, L)), xi, K, qfrac[0])
                    obj_label = minimize_objective_quardratic(D, y.reshape(L, L), xi, K, qfrac[0])
            elif args.g_type == "excess":
                D, outputs, xi, K = D.reshape(L, L), outputs.reshape(L, L), xi.flatten(), K.flatten()
                obj_predict = minimize_objective_excess(D, outputs.reshape(L, L), xi, K)
                obj_predict_zero_pair = minimize_objective_excess(D, zero_refine(zero_refine(outputs).reshape(L, L)), xi, K)
                obj_label = minimize_objective_excess(D, y.reshape(L, L), xi, K)
            else:
                print("no such d_type {args.g_type}")

            minimize_objective_predict += obj_predict
            minimize_objective_predict_zero_pair += obj_predict_zero_pair
            minimize_objective__label += obj_label

    predict_score /= inference_count
    predict_score_zero_pair /= inference_count
    predict_non_zero_count /= inference_count
    label_score /= inference_count
    label_non_zero_count /= inference_count
    mse_loss /= inference_count
    f1 /= inference_count
    tn /= inference_count
    fp /= inference_count
    fn /= inference_count
    tp /= inference_count
    f1_zero_refine /= inference_count
    tn_zero_refine /= inference_count
    fp_zero_refine /= inference_count
    fn_zero_refine /= inference_count
    tp_zero_refine /= inference_count
    minimize_objective_predict /= inference_count
    minimize_objective_predict_zero_pair /= inference_count
    minimize_objective__label /= inference_count
    return {
            "inference_step": args.inference_step,
            "predict_score": predict_score.item(),
            "predict_score_zero_pair": predict_score_zero_pair,
            "predict_non_zero_count": predict_non_zero_count,
            "label_score": label_score.item(),
            "label_non_zero_count": label_non_zero_count,
            "mse_loss": mse_loss,
            "zero_threshold": zero_threshold,
            "f1_score": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "f1_zero_refine": f1_zero_refine, 
            "tn_zero_refine": tn_zero_refine, 
            "fp_zero_refine": fp_zero_refine, 
            "fn_zero_refine": fn_zero_refine, 
            "tp_zero_refine": tp_zero_refine,
            "total_value": y.numel(),
            "minimize_objective_predict": minimize_objective_predict.item(),
            "minimize_objective_predict_zero_pair": minimize_objective_predict_zero_pair.item(),
            "minimize_objective__label": minimize_objective__label.item()
            }

def train_stage2_NN(train_loader, val_loader, test_loader, model, stage1_model, criterion, optimizer, args, task_name, device):
    global_step = 0
    train_log = {"epoch":[], "step":[], "train_loss":[], "eval_loss":[], "test_loss":[],
                 "min_eval_loss": float('inf'), 
                 "min_cost_objective": float('inf'),
                 "max_f1_score": float(0),
                 "min_infesibility": float('inf'),
                "inference_log":[]}
    train_bar = tqdm(total=args.train_step)
    while global_step <= args.train_step:
        model.train()
        for data_batch in train_loader:
            if args.exnt:
                x, y, K, mu, sigma, D, L, int_y = (data_batch["x"].to(device), 
                       data_batch["y"].to(device), 
                       data_batch["x_origin"]["K"].to(device),
                       data_batch["x_origin"]["mus"].to(device),
                       data_batch["x_origin"]["sigmas"].to(device),
                       data_batch["x_origin"]["D"].to(device),
                       data_batch["x_origin"]["L"],
                       data_batch["int_y"].to(device))
                stage1_model.eval()
                stage1_output = stage1_model(x)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                stage2_input = torch.cat((zero_refine_s1_output, x), dim=1) # cat with zero refined inputs
                
                optimizer.zero_grad()
                outputs = model(stage2_input)
                
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                
                stage2_outputs = outputs.view(-1, 2)
                # print(outputs.shape, action_label.shape)
                loss = criterion(outputs, action_label, K, mu, sigma, D, L, args, stage2=True)
            else:
                x, y, K, xi, D, L, int_y, qfrac = (data_batch["x"].to(device), 
                        data_batch["y"].to(device), 
                        data_batch["x_origin"]["K"].to(device),
                        data_batch["x_origin"]["xi"].to(device),
                        data_batch["x_origin"]["D"].to(device),
                        data_batch["x_origin"]["L"],
                        data_batch["int_y"].to(device),
                        data_batch["x_origin"]["qfrac"].to(device)
                        )
                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output = stage1_model(x)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                stage2_input = torch.cat((zero_refine_s1_output, x), dim=1) # cat with zero refined inputs
                
                optimizer.zero_grad()
                outputs = model(stage2_input)
                
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                # one_hot_action_label = torch.nn.functional.one_hot(action_label, num_classes=2).view(-1,2).float()
                # print(one_hot_action_label.shape, stage2_outputs.shape)
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(stage2_outputs, action_label, K, xi, D, L, args, qfrac=qfrac[0],stage2=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_bar.update(1)
            train_bar.set_postfix(loss = loss.item())

            if global_step % args.eval_step == 0 or global_step == 1: # eval model when reach certain step
                val_loss = test_stage2_NN(val_loader, model, criterion, device, args, stage1_model)
                test_loss = test_stage2_NN(test_loader, model, criterion, device, args, stage1_model)
                train_log["epoch"].append(global_step / len(train_loader))
                train_log["step"].append(global_step)
                train_log["train_loss"].append(loss.item())
                train_log["eval_loss"].append(val_loss)
                train_log["test_loss"].append(test_loss)
                inference_log = inference_stage2_NN(model, val_loader.dataset, device, stage1_model=stage1_model,
                                                zero_threshold=float("1e-2"), args=args, inference_step=args.inference_step)
                train_log["inference_log"].append(inference_log)
                if "eval_loss" in args.model_selection_strategy:
                    if val_loss < train_log["min_eval_loss"]: # if the eval loss is minimum, save the model as the best model.
                        train_log["min_eval_loss"] = val_loss
                        train_log["best_step_eval"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_eval_loss.pth"))
                if "f1" in args.model_selection_strategy:
                    if inference_log["f1_score"] > train_log["max_f1_score"]: # if the f1_score is max, save the model as the best model.
                        train_log["max_f1_score"] = inference_log["f1_score"]
                        train_log["best_step_f1"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_f1.pth"))
                if "infesibility" in args.model_selection_strategy:
                    if inference_log["predict_score"] < train_log["min_infesibility"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_infesibility"] = inference_log["predict_score"]
                        train_log["best_step_infesibility"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_infesibility.pth")) 
                if "cost_obj" in args.model_selection_strategy:
                    if inference_log["minimize_objective_predict"] < train_log["min_cost_objective"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_cost_objective"] = inference_log["minimize_objective_predict"]
                        train_log["best_step_cost_obj"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_cost_obj.pth"))          
            if global_step >= args.train_step: 
                break
        train_bar.close
    return train_log

def test_stage2_NN(tst_loader, model, criterion, device, args, stage1_model):
    model.eval()
    tst_loss = 0.0
    eval_loop = tqdm((tst_loader), total = len(tst_loader))
    with torch.no_grad():
        for data_batch in eval_loop:
            if args.exnt:
                x, y, K, mu, sigma, D, L, int_y, y_float_obj, y_int_obj = (data_batch["x"].to(device), 
                       data_batch["y"].to(device), 
                       data_batch["x_origin"]["K"].to(device),
                       data_batch["x_origin"]["mus"].to(device),
                       data_batch["x_origin"]["sigmas"].to(device),
                       data_batch["x_origin"]["D"].to(device),
                       data_batch["x_origin"]["L"],
                       data_batch["int_y"].to(device),
                       data_batch["x_origin"]["float_obj"].to(device),
                       data_batch["x_origin"]["int_obj"].to(device))
                stage1_model.eval()
                stage1_output = stage1_model(x)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                stage2_input = torch.cat((zero_refine_s1_output, x), dim=1) # cat with zero refined inputs
                
                outputs = model(stage2_input)
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                
                stage2_outputs = outputs.view(-1, 2)
                loss = criterion(outputs, action_label, K, mu, sigma, D, L, args, stage2=True)
            else:
                x, y, K, xi, D, L, int_y, qfrac = (data_batch["x"].to(device), 
                        data_batch["y"].to(device), 
                        data_batch["x_origin"]["K"].to(device),
                        data_batch["x_origin"]["xi"].to(device),
                        data_batch["x_origin"]["D"].to(device),
                        data_batch["x_origin"]["L"],
                        data_batch["int_y"].to(device),
                        data_batch["x_origin"]["qfrac"].to(device)
                        )
                stage1_model.eval()
                stage1_output = stage1_model(x)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                stage2_input = torch.cat((zero_refine_s1_output, x), dim=1) # cat with zero refined inputs
                outputs = model(stage2_input)

                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                # one_hot_action_label = torch.nn.functional.one_hot(action_label, num_classes=2).view(-1,2).float()
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(stage2_outputs, action_label, K, xi, D, L, args, qfrac=qfrac[0], stage2=True)
            tst_loss += loss.item()
            eval_loop.set_postfix(loss=loss.item())
    tst_loss /= len(tst_loader)
    eval_loop.close
    return tst_loss

def inference_stage2_NN(model, tst_dataset, device, args, zero_threshold, stage1_model, inference_step=-1):
    # inference over dataset not dataloader, process samples one by one
    model.eval()
    loop = tqdm((tst_dataset), total = len(tst_dataset))
    predict_score = 0
    predict_score_zero_pair = 0
    predict_non_zero_count = 0
    label_score = 0
    label_non_zero_count = 0
    inference_count = 0
    mse_loss = 0
    f1, tn, fp, fn, tp = 0, 0, 0, 0, 0
    f1_zero_refine, tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = 0, 0, 0, 0, 0
    minimize_objective_predict = 0
    minimize_objective_predict_zero_pair = 0
    minimize_objective__label = 0
    with torch.no_grad():
        for idx, data in enumerate(loop):
            # model output
            if args.exnt:
                x, y, K, mu, sigma, D, L, int_y = (data["x"].to(device), 
                       data["y"].to(device), 
                       torch.tensor(data["x_origin"]["K"]).to(device),
                       torch.tensor(data["x_origin"]["mus"]).to(device),
                       torch.tensor(data["x_origin"]["sigmas"]).to(device),
                       torch.tensor(data["x_origin"]["D"]).to(device),
                       data["x_origin"]["L"],
                       data["int_y"].to(device))
                stage1_model.eval()
                stage1_output = stage1_model(x.unsqueeze(0))
                zero_refine_s1_output = zero_refine(stage1_output.reshape(L, L))
                stage2_input = torch.cat((zero_refine_s1_output.flatten().unsqueeze(0), x.unsqueeze(0)), dim=1) # cat with zero refined inputs
                outputs = model(stage2_input)
            else:
                x, y, K, xi, D, L, int_y = (data["x"].to(device), 
                        data["y"].to(device), 
                        torch.tensor(data["x_origin"]["K"]).to(device),
                        torch.tensor(data["x_origin"]["xi"]).to(device),
                        torch.tensor(data["x_origin"]["D"]).to(device),
                        data["x_origin"]["L"],
                        data["int_y"].to(device))
                
                stage1_model.eval()
                stage1_output = stage1_model(x.unsqueeze(0))
                zero_refine_s1_output = zero_refine(stage1_output.reshape(L, L))
                stage2_input = torch.cat((zero_refine_s1_output.flatten().unsqueeze(0), x.unsqueeze(0)), dim=1) # cat with zero refined inputs
                outputs = model(stage2_input) # onehot (BS*L*L,2)
            predicted_actions = torch.argmax(outputs, dim=-1) #BLL,

            refined_stage1_solution = rounded_by_binary_action(zero_refine_s1_output.flatten(), predicted_actions).reshape(L, L) # use the refined output to do prediction.


            predict_score += infeasibility(zero_refine_s1_output, K) # input 
            predict_non_zero_count += torch.sum(refined_stage1_solution > zero_threshold).item()
            label_score += infeasibility(int_y.unsqueeze(0), K)
            label_non_zero_count += torch.sum(int_y > zero_threshold).item()
            mse_loss += nn.MSELoss()(refined_stage1_solution.flatten(), int_y.flatten()).item()

            # f1_score
            solution_binary = torch.where(int_y < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_binary = torch.where(refined_stage1_solution < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_zero_refine_binary = torch.where(zero_refine(refined_stage1_solution).flatten().unsqueeze(0) < zero_threshold, torch.tensor(0), torch.tensor(1))
            f1 += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_binary.flatten().cpu().numpy())
            conf_matrix = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn, fp, fn, tp = tuple( x + y for x , y in zip((tn, fp, fn, tp), conf_matrix))

            f1_zero_refine += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1])
            conf_matrix_zero_refine = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = tuple( x + y for x , y in zip((tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine), conf_matrix_zero_refine))

            inference_count += 1
            if inference_step != -1 and idx >= inference_step:
                break

            # minimize_objective_
            if args.g_type == "quadratic":
                if args.exnt:
                    # D, outputs, mu, sigma, K = D.reshape(L, L), outputs.reshape(L, L), mu.flatten(), sigma.flatten(), K.flatten()
                    obj_predict = minimize_objective_quardratic_exnt(D, zero_refine_s1_output,  mu, sigma, K)
                    obj_predict_zero_pair = minimize_objective_quardratic_exnt(D, refined_stage1_solution,  mu, sigma, K)
                    obj_label = minimize_objective_quardratic_exnt(D, int_y.reshape(L, L),  mu, sigma, K)
                else:
                    fqurd = torch.tensor(data["x_origin"]["qfrac"]).to(device)
                    obj_predict = minimize_objective_quardratic(D, zero_refine_s1_output, xi, K, fqurd)
                    obj_predict_zero_pair = minimize_objective_quardratic(D, zero_refine(refined_stage1_solution), xi, K, fqurd)
                    obj_label = minimize_objective_quardratic(D, int_y.reshape(L, L), xi, K, fqurd)
            elif args.g_type == "excess":
                obj_predict = minimize_objective_excess(D, zero_refine_s1_output, xi, K)
                obj_predict_zero_pair = minimize_objective_excess(D, zero_refine(refined_stage1_solution), xi, K)
                obj_label = minimize_objective_excess(D, int_y.reshape(L, L), xi, K)
            else:
                print("no such d_type {args.g_type}")
            minimize_objective_predict += obj_predict
            minimize_objective_predict_zero_pair += obj_predict_zero_pair
            minimize_objective__label += obj_label

    predict_score /= inference_count
    predict_score_zero_pair /= inference_count
    predict_non_zero_count /= inference_count
    label_score /= inference_count
    label_non_zero_count /= inference_count
    mse_loss /= inference_count
    f1 /= inference_count
    tn /= inference_count
    fp /= inference_count
    fn /= inference_count
    tp /= inference_count
    f1_zero_refine /= inference_count
    tn_zero_refine /= inference_count
    fp_zero_refine /= inference_count
    fn_zero_refine /= inference_count
    tp_zero_refine /= inference_count
    minimize_objective_predict /= inference_count
    minimize_objective_predict_zero_pair /= inference_count
    minimize_objective__label /= inference_count
    return {
            "inference_step": args.inference_step,
            "predict_score": predict_score.item(),
            "predict_score_zero_pair": predict_score_zero_pair,
            "predict_non_zero_count": predict_non_zero_count,
            "label_score": label_score.item(),
            "label_non_zero_count": label_non_zero_count,
            "mse_loss": mse_loss,
            "zero_threshold": zero_threshold,
            "f1_score": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "f1_zero_refine": f1_zero_refine, 
            "tn_zero_refine": tn_zero_refine, 
            "fp_zero_refine": fp_zero_refine, 
            "fn_zero_refine": fn_zero_refine, 
            "tp_zero_refine": tp_zero_refine,
            "total_value": y.numel(),
            "minimize_objective_predict": minimize_objective_predict.item(),
            "minimize_objective_predict_zero_pair": minimize_objective_predict_zero_pair.item(),
            "minimize_objective__label": minimize_objective__label.item()
            }

def train_stage2_GNN(train_loader, val_loader, test_loader, model, stage1_model, criterion, optimizer, args, task_name, device):
    global_step = 0
    train_log = {"epoch":[], "step":[], "train_loss":[], "eval_loss":[], "test_loss":[],
                 "min_eval_loss": float('inf'), 
                 "min_cost_objective": float('inf'),
                 "max_f1_score": float(0),
                 "min_infesibility": float('inf'),
                "inference_log":[]}
    train_bar = tqdm(total=args.train_step)
    while global_step <= args.train_step:
        model.train()
        for data_batch in train_loader:
            data, y = data_batch
            if args.exnt:
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                )
                L, K, mu, sigma, D = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["mus"])).to(device),
                    torch.tensor(np.array(data.features["sigmas"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                )

                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))
                # print(stage1_output.shape)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                # print(zero_refine_s1_output.shape, nn_x.shape)
                nn_x = nn_x.reshape(zero_refine_s1_output.shape[0], -1)
                stage2_input = torch.cat((zero_refine_s1_output, nn_x), dim=1) # cat with zero refined inputs
                
                optimizer.zero_grad()
                outputs = model(stage2_input)
                
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(outputs, action_label, K, mu, sigma, D, L, args, stage2=True)
            else:
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                )
                
                L, K, xi, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    torch.tensor(np.array(data.features["qfrac"])).to(device)
                )
                
                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))
                # print(stage1_output.shape)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                # print(zero_refine_s1_output.shape, nn_x.shape)
                nn_x = nn_x.reshape(zero_refine_s1_output.shape[0], -1)
                stage2_input = torch.cat((zero_refine_s1_output, nn_x), dim=1) # cat with zero refined inputs
                
                optimizer.zero_grad()
                outputs = model(stage2_input)
            
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(stage2_outputs, action_label, K, xi, D, L, args, stage2=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            train_bar.update(1)
            train_bar.set_postfix(loss = loss.item())

            if global_step % args.eval_step == 0 or global_step == 1: # eval model when reach certain step
                val_loss = test_stage2_GNN(val_loader, model, criterion, device, args, stage1_model)
                test_loss = test_stage2_GNN(test_loader, model, criterion, device, args, stage1_model)
                train_log["epoch"].append(global_step / len(train_loader))
                train_log["step"].append(global_step)
                train_log["train_loss"].append(loss.item())
                train_log["eval_loss"].append(val_loss)
                train_log["test_loss"].append(test_loss)
                inference_log = inference_stage2_GNN(model, val_loader.dataset, device, stage1_model=stage1_model,
                                                zero_threshold=float("1e-2"), args=args, inference_step=args.inference_step)
                train_log["inference_log"].append(inference_log)
                if "eval_loss" in args.model_selection_strategy:
                    if val_loss < train_log["min_eval_loss"]: # if the eval loss is minimum, save the model as the best model.
                        train_log["min_eval_loss"] = val_loss
                        train_log["best_step_eval"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_eval_loss.pth"))
                if "f1" in args.model_selection_strategy:
                    if inference_log["f1_score"] > train_log["max_f1_score"]: # if the f1_score is max, save the model as the best model.
                        train_log["max_f1_score"] = inference_log["f1_score"]
                        train_log["best_step_f1"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_f1.pth"))
                if "infesibility" in args.model_selection_strategy:
                    if inference_log["predict_score"] < train_log["min_infesibility"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_infesibility"] = inference_log["predict_score"]
                        train_log["best_step_infesibility"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_infesibility.pth")) 
                if "cost_obj" in args.model_selection_strategy:
                    if inference_log["minimize_objective_predict"] < train_log["min_cost_objective"]: # if the infesibility is minimum, save the model as the best model.
                        train_log["min_cost_objective"] = inference_log["minimize_objective_predict"]
                        train_log["best_step_cost_obj"] = global_step
                        task_save_path = os.path.join(args.checkpoint_path, task_name)
                        if not os.path.exists(task_save_path):
                            os.makedirs(task_save_path)
                        torch.save(model.state_dict(), os.path.join(task_save_path, f"best_{args.model_type}_cost_obj.pth"))          
            if global_step >= args.train_step: 
                break
        train_bar.close
    return train_log

def test_stage2_GNN(tst_loader, model, criterion, device, args, stage1_model):
    model.eval()
    tst_loss = 0.0
    eval_loop = tqdm((tst_loader), total = len(tst_loader))
    with torch.no_grad():
        for data_batch in eval_loop:
            data, y = data_batch
            if args.exnt:
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                    )
                L, K, mu, sigma, D, qfrac = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["mus"])).to(device),
                    torch.tensor(np.array(data.features["sigmas"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device)
                    )

                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))
                # print(stage1_output.shape)
                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                # print(zero_refine_s1_output.shape, nn_x.shape)
                nn_x = nn_x.reshape(zero_refine_s1_output.shape[0], -1)
                stage2_input = torch.cat((zero_refine_s1_output, nn_x), dim=1) # cat with zero refined inputs
                
                outputs = model(stage2_input)
            
                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(outputs, action_label, K, mu, sigma, D, L, args, stage2=True)
            else:  
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                )
                L, K, xi, D = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device)
                )
                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))

                zero_refine_s1_output = zero_refine_batch(stage1_output.reshape(-1, L[0], L[0])).reshape(-1, L[0]**2)
                nn_x = nn_x.reshape(zero_refine_s1_output.shape[0], -1)
                stage2_input = torch.cat((zero_refine_s1_output, nn_x), dim=1) # cat with zero refined inputs

                outputs = model(stage2_input)

                action_label = (int_y.flatten() - zero_refine_s1_output.flatten() > args.round_threshold).long()
                stage2_outputs = outputs.view(-1, 2)

                loss = criterion(stage2_outputs, action_label, K, xi, D, L, args, stage2=True)
            tst_loss += loss.item()
            eval_loop.set_postfix(loss=loss.item())
    tst_loss /= len(tst_loader)
    eval_loop.close
    return tst_loss

def inference_stage2_GNN(model, tst_dataset, device, args, zero_threshold, stage1_model, inference_step=-1):
    # inference over dataset not dataloader, process samples one by one
    model.eval()
    tst_loader =  GraphDataloader(tst_dataset, batch_size=1, shuffle=False)
    loop = tqdm((tst_loader), total = len(tst_dataset))
    predict_score = 0
    predict_score_zero_pair = 0
    predict_non_zero_count = 0
    label_score = 0
    label_non_zero_count = 0
    inference_count = 0
    mse_loss = 0
    f1, tn, fp, fn, tp = 0, 0, 0, 0, 0
    f1_zero_refine, tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = 0, 0, 0, 0, 0
    minimize_objective_predict = 0
    minimize_objective_predict_zero_pair = 0
    minimize_objective__label = 0
    with torch.no_grad():
        for idx, data in enumerate(loop):
            # model output
            data, y = data
            if args.exnt:
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                    )
                L, K, mu, sigma, D = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["mus"])).to(device),
                    torch.tensor(np.array(data.features["sigmas"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                    )

                # get stage1 model output to refine
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))
                # print(stage1_output.shape)
                zero_refine_s1_output = zero_refine(stage1_output.reshape(L, L))
                # print(zero_refine_s1_output.shape, nn_x.shape)
                stage2_input = torch.cat((zero_refine_s1_output.flatten().unsqueeze(0), nn_x.flatten().unsqueeze(0)), dim=1) # cat with zero refined inputs
                
                outputs = model(stage2_input)
            else:
                x, edge_index, edge_attr, y, nn_x, int_y = (
                    data.x.to(device),
                    data.edge_index.to(device),
                    data.edge_attr.to(device),
                    y.to(device),
                    data.features_vector.to(device),
                    data.int_y.to(device)
                )
                L, K, xi, D = (
                    data.features["L"].to(device),
                    torch.tensor(np.array(data.features["K"])).to(device),
                    torch.tensor(np.array(data.features["xi"])).to(device),
                    torch.tensor(np.array(data.features["D"])).to(device),
                )
                # get stage1 model output to refine

                # start_time = time.time()
                stage1_model.eval()
                stage1_output =  stage1_model(x, edge_index, edge_attr, data.batch.to(device))
                
                zero_refine_s1_output = zero_refine(stage1_output.reshape(L, L))
                # time_step1 = time.time()
                stage2_input = torch.cat((zero_refine_s1_output.flatten().unsqueeze(0), nn_x.unsqueeze(0)), dim=1) # cat with zero refined inputs
                outputs = model(stage2_input) # onehot (BS*L*L,2)

            predicted_actions = torch.argmax(outputs, dim=-1) #BLL,

            # this is the final refined output solution in shape 1,L,L
            refined_stage1_solution = rounded_by_binary_action(zero_refine_s1_output.flatten(), predicted_actions).reshape(L, L) # use the refined output to do prediction.
            
            # end_time = time.time()
            # elapsed_time_int = end_time - start_time
            # elapsed_time_float = time_step1 - start_time
            # print(f"!!!!!!!!!!!!! took {elapsed_time_float:.6f} seconds to run.")
            # print(f"!!!!!!!!!!!!! took {elapsed_time_int:.6f} seconds to run.")
            # exit()


            predict_score += infeasibility(zero_refine_s1_output, K) # input 
            predict_non_zero_count += torch.sum(refined_stage1_solution > zero_threshold).item()
            label_score += infeasibility(int_y.unsqueeze(0), K)
            label_non_zero_count += torch.sum(int_y > zero_threshold).item()
            mse_loss += nn.MSELoss()(refined_stage1_solution.flatten(), int_y.flatten()).item()

            # f1_score
            solution_binary = torch.where(int_y < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_binary = torch.where(refined_stage1_solution.flatten().unsqueeze(0) < zero_threshold, torch.tensor(0), torch.tensor(1))
            output_zero_refine_binary = torch.where(zero_refine(refined_stage1_solution).flatten().unsqueeze(0) < zero_threshold, torch.tensor(0), torch.tensor(1))
            f1 += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_binary.flatten().cpu().numpy())
            conf_matrix = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn, fp, fn, tp = tuple( x + y for x , y in zip((tn, fp, fn, tp), conf_matrix))

            f1_zero_refine += f1_score(solution_binary.flatten().cpu().numpy(), 
                           output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1])
            conf_matrix_zero_refine = confusion_matrix(solution_binary.flatten().cpu().numpy(), 
                                              output_zero_refine_binary.flatten().cpu().numpy(), labels=[0, 1]).ravel()
            tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine = tuple( x + y for x , y in zip((tn_zero_refine, fp_zero_refine, fn_zero_refine, tp_zero_refine), conf_matrix_zero_refine))

            inference_count += 1
            if inference_step != -1 and idx >= inference_step:
                break

            # minimize_objective_
            if args.g_type == "quadratic":
                if args.exnt:
                        # D, outputs, mu, sigma, K = D.reshape(L, L), outputs.reshape(L, L), mu.flatten(), sigma.flatten(), K.flatten()
                    obj_predict = minimize_objective_quardratic_exnt(D, zero_refine_s1_output,  mu, sigma, K)
                    obj_predict_zero_pair = minimize_objective_quardratic_exnt(D, refined_stage1_solution,  mu, sigma, K)
                    obj_label = minimize_objective_quardratic_exnt(D, int_y.reshape(L, L),  mu, sigma, K)
                else:
                    qfrac = torch.tensor(np.array(data.features["qfrac"]))[0].to(device)
                    obj_predict = minimize_objective_quardratic(D, zero_refine_s1_output, xi, K, qfrac=qfrac)
                    obj_predict_zero_pair = minimize_objective_quardratic(D, zero_refine(refined_stage1_solution), xi, K, qfrac=qfrac)
                    obj_label = minimize_objective_quardratic(D, int_y.reshape(L, L), xi, K, qfrac=qfrac)
            elif args.g_type == "excess":
                obj_predict = minimize_objective_excess(D, zero_refine_s1_output, xi, K)
                obj_predict_zero_pair = minimize_objective_excess(D, zero_refine(refined_stage1_solution), xi, K)
                obj_label = minimize_objective_excess(D, int_y.reshape(L, L), xi, K)
            else:
                print("no such d_type {args.g_type}")
            
            minimize_objective_predict += obj_predict
            minimize_objective_predict_zero_pair += obj_predict_zero_pair
            minimize_objective__label += obj_label

    predict_score /= inference_count
    predict_score_zero_pair /= inference_count
    predict_non_zero_count /= inference_count
    label_score /= inference_count
    label_non_zero_count /= inference_count
    mse_loss /= inference_count
    f1 /= inference_count
    tn /= inference_count
    fp /= inference_count
    fn /= inference_count
    tp /= inference_count
    f1_zero_refine /= inference_count
    tn_zero_refine /= inference_count
    fp_zero_refine /= inference_count
    fn_zero_refine /= inference_count
    tp_zero_refine /= inference_count
    minimize_objective_predict /= inference_count
    minimize_objective_predict_zero_pair /= inference_count
    minimize_objective__label /= inference_count
    return {
            "inference_step": args.inference_step,
            "predict_score": predict_score.item(),
            "predict_score_zero_pair": predict_score_zero_pair,
            "predict_non_zero_count": predict_non_zero_count,
            "label_score": label_score.item(),
            "label_non_zero_count": label_non_zero_count,
            "mse_loss": mse_loss,
            "zero_threshold": zero_threshold,
            "f1_score": f1,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "f1_zero_refine": f1_zero_refine, 
            "tn_zero_refine": tn_zero_refine, 
            "fp_zero_refine": fp_zero_refine, 
            "fn_zero_refine": fn_zero_refine, 
            "tp_zero_refine": tp_zero_refine,
            "total_value": y.numel(),
            "minimize_objective_predict": minimize_objective_predict.item(),
            "minimize_objective_predict_zero_pair": minimize_objective_predict_zero_pair.item(),
            "minimize_objective__label": minimize_objective__label.item()
            }

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

