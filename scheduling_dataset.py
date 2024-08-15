import torch
from torch.utils.data import Dataset
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import DataLoader
from  torch_geometric.loader.dataloader import DataLoader as GDataloader
from tqdm import tqdm
import numpy as np
from utils import infeasibility

class SchedulingDataset(Dataset):
    def __init__(self, data, kxi_scale, d_scale):
        super(SchedulingDataset, self).__init__()
        self.data = data
        self.kxi_scale = kxi_scale
        self.d_scale = d_scale
        if len(self.data["int_solution"]) == 0:
            self.data.pop("int_solution")

    def __getitem__(self, idx):
        features = {}
        #L, T, xi_t, K, D
        for key in self.data.keys():
            features[key] = self.data[key][idx]
        solution = features["solution"]

        features_vector = torch.cat((self.scale_normalize(torch.tensor(features["xi"].flatten()), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["K"].flatten()), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["D"].flatten()), self.d_scale)), dim=0).float()

        solution_vector = torch.tensor(solution.flatten()).float()

        if "int_solution" not in features.keys():
            int_solution_vector = solution_vector
        else:
            int_solution_vector = torch.tensor(features["int_solution"]).float().flatten()
        return {
                "x": features_vector, 
                "y": solution_vector, 
                "int_y": int_solution_vector,
                "x_origin": features, 
                "y_origin": solution,
                "kxi_scale": self.kxi_scale,
                "d_scale": self.d_scale,
        }

    def __len__(self):
        return len(self.data["L"])
    
    def l2_normalize(self, tensor):
        norm = tensor.norm()
        normalized_tensor = tensor / norm
        return normalized_tensor, norm

    def scale_normalize(self, tensor, scale):
        return tensor / scale

class SchedulingGraphDataset(Dataset):
    def __init__(self, data, kxi_scale, d_scale):
        super(SchedulingGraphDataset, self).__init__()
        self.data = data
        self.kxi_scale = kxi_scale
        self.d_scale = d_scale
        if len(self.data["int_solution"]) == 0:
            self.data.pop("int_solution")

    def __getitem__(self, idx):
        features = {}
        for key in self.data.keys():
            features[key] = self.data[key][idx]
        solution = features["solution"]
        node_features = torch.cat((self.scale_normalize(torch.tensor(features["xi"].reshape(-1, 1)), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["K"].reshape(-1, 1)), self.kxi_scale)), dim=1)
        edge_features = self.scale_normalize(torch.tensor(features["D"]), self.d_scale)

        features_vector = torch.cat((self.scale_normalize(torch.tensor(features["xi"].flatten()), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["K"].flatten()), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["D"].flatten()), self.d_scale)), dim=0).float()

        solution_vector = torch.tensor(solution.flatten()).float()
        edge_index = torch.nonzero(edge_features, as_tuple=False).t()
        if "int_solution" not in features.keys():
            int_solution_vector = solution_vector
        else:
            int_solution_vector = torch.tensor(features["int_solution"]).float().flatten()

        data = Data(
            x=node_features.float(), 
            edge_index=edge_index.long(), 
            edge_attr=edge_features.float().flatten(), 
            features= features,
            int_y = int_solution_vector,
            kxi_scale = self.kxi_scale,
            d_scale = self.d_scale,
            features_vector = features_vector
        )

        return data, solution_vector

    def __len__(self):
        return len(self.data["L"])

    def l2_normalize(self, tensor):
        norm = tensor.norm()
        normalized_tensor = tensor / norm
        return normalized_tensor

    def scale_normalize(self, tensor, scale):
        return tensor / scale

class SchedulingExntDataset(Dataset):
    def __init__(self, data, kxi_scale, d_scale):
        super(SchedulingExntDataset, self).__init__()
        self.data = data
        self.kxi_scale = kxi_scale
        self.d_scale = d_scale
        if len(self.data["int_solution"]) == 0:
            self.data.pop("int_solution")

    def __getitem__(self, idx):
        features = {}
        #L, T, xi_t, K, D
        for key in self.data.keys():
            if len(self.data[key]) != 0:
                features[key] = self.data[key][idx]

        solution = features["solution"]

        features_vector = torch.cat((self.l2_normalize(torch.tensor(np.array(features["mus"])).flatten())[0],
                                     self.l2_normalize(torch.tensor(np.array(features["sigmas"])).flatten())[0],
                                     self.scale_normalize(torch.tensor(features["K"]).flatten(), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["D"]).flatten(), self.d_scale)), dim=0).float()

        solution_vector = torch.tensor(solution.flatten()).float()

        if "int_solution" not in features.keys():
            int_solution_vector = solution_vector
        else:
            int_solution_vector = torch.tensor(features["int_solution"]).float().flatten()
        return {
                "x": features_vector, 
                "y": solution_vector, 
                "int_y": int_solution_vector,
                "x_origin": features, 
                "y_origin": solution,
                "kxi_scale": self.kxi_scale,
                "d_scale": self.d_scale,
        }

    def __len__(self):
        return len(self.data["L"])
    
    def l2_normalize(self, tensor):
        norm = tensor.norm()
        normalized_tensor = tensor / norm
        return normalized_tensor, norm

    def scale_normalize(self, tensor, scale):
        return tensor / scale

class SchedulingGraphExntDataset(Dataset):
    def __init__(self, data, kxi_scale, d_scale):
        super(SchedulingGraphExntDataset, self).__init__()
        self.data = data
        self.kxi_scale = kxi_scale
        self.d_scale = d_scale
        if len(self.data["int_solution"]) == 0:
            self.data.pop("int_solution")

    def __getitem__(self, idx):
        features = {}
        for key in self.data.keys():
            if len(self.data[key]) != 0:
                features[key] = self.data[key][idx]
        solution = features["solution"]
        node_features = torch.cat((self.l2_normalize(torch.tensor(features["mus"].reshape(-1, 1))),
                                    self.l2_normalize(torch.tensor(features["sigmas"].reshape(-1, 1))),
                                     self.scale_normalize(torch.tensor(features["K"].reshape(-1, 1)), self.kxi_scale)), dim=1)
        edge_features = self.scale_normalize(torch.tensor(features["D"]), self.d_scale)

        features_vector = torch.cat((self.l2_normalize(torch.tensor(features["mus"]).flatten()),
                                     self.l2_normalize(torch.tensor(features["sigmas"]).flatten()),
                                     self.scale_normalize(torch.tensor(features["K"]).flatten(), self.kxi_scale),
                                     self.scale_normalize(torch.tensor(features["D"]).flatten(), self.d_scale)), dim=0).float()

        solution_vector = torch.tensor(solution.flatten()).float()
        edge_index = torch.nonzero(edge_features, as_tuple=False).t()
        if "int_solution" not in features.keys():
            int_solution_vector = solution_vector
        else:
            int_solution_vector = torch.tensor(features["int_solution"]).float().flatten()

        data = Data(
            x=node_features.float(), 
            edge_index=edge_index.long(), 
            edge_attr=edge_features.float().flatten(), 
            features= features,
            int_y = int_solution_vector,
            kxi_scale = self.kxi_scale,
            d_scale = self.d_scale,
            features_vector = features_vector
        )

        return data, solution_vector

    def __len__(self):
        return len(self.data["L"])

    def l2_normalize(self, tensor):
        norm = tensor.norm()
        normalized_tensor = tensor / norm
        return normalized_tensor

    def scale_normalize(self, tensor, scale):
        return tensor / scale       
if __name__ == "__main__":
    # test script
    with open("./data/X_5120_L8T24_quadratic_zerosFalse.pkl", 'rb') as file:
        data_X = pickle.load(file)

    with open("./data/y_5120_L8T24_quadratic_zerosFalse.pkl", 'rb') as file:
        data_y = pickle.load(file)

    ds = SchedulingGraphDataset(data_X, data_y, 500)
    dl = GDataloader(ds, batch_size=2, shuffle=True)
    for d in dl:
        print(d.y)
        break
