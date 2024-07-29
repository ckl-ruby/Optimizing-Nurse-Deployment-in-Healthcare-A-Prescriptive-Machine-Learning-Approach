import torch
from torch.utils.data import Dataset
import pickle

class SchedulingDataset(Dataset):
    def __init__(self, data, labels):
        super(SchedulingDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, idx):
        features = {}
        #L, T, xi_t, K, D
        for key in self.data.keys():
            features[key] = self.data[key][idx]
        solution = self.labels["solution"][idx]
        features_vector = torch.cat((torch.tensor(features["xi_t"].flatten()),
                                     torch.tensor(features["K"].flatten()),
                                     torch.tensor(features["D"].flatten())), dim=0)
        solution_vector = torch.tensor(solution.flatten())
        return features_vector, solution_vector, features, solution

    def __len__(self):
        return self.labels["solution"].shape[0]


if __name__ == "__main__":
    # test script
    with open("./data/data_X_1024.pkl", 'rb') as file:
        data_X = pickle.load(file)

    with open("./data/data_y_1024.pkl", 'rb') as file:
        data_y = pickle.load(file)

    ds = SchedulingDataset(data_X, data_y)
    feature, label = ds[0]
    print(feature, label)
    print(feature.shape, label.shape)
