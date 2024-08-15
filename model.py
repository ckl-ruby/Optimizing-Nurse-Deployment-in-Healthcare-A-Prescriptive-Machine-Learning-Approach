import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SchedulingNN(nn.Module):
    def __init__(self, input_dim, d_model, out_dim, mid_layer):
        super(SchedulingNN, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.out_dim = out_dim

        self.network = nn.ModuleList()
        self.network.append(nn.Linear(self.input_dim, self.d_model))
        self.network.append(nn.ReLU())
        self.network.append(nn.BatchNorm1d(self.d_model))
        for i in range(mid_layer):
            self.network.append(nn.Linear(self.d_model, self.d_model))
            self.network.append(nn.ReLU())
            self.network.append(nn.BatchNorm1d(self.d_model))
        self.network.append(nn.Linear(self.d_model, self.out_dim))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        return x

class StageTwoNN(nn.Module):
    def __init__(self, input_dim, d_model, out_dim, mid_layer):
        super(StageTwoNN, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.out_dim = out_dim

        self.network = nn.ModuleList()
        self.network.append(nn.Linear(self.input_dim , self.d_model))
        self.network.append(nn.ReLU())
        self.network.append(nn.BatchNorm1d(self.d_model))
        for i in range(mid_layer):
            self.network.append(nn.Linear(self.d_model, self.d_model))
            self.network.append(nn.ReLU())
            self.network.append(nn.BatchNorm1d(self.d_model))
        self.network.append(nn.Linear(self.d_model, 2 * self.out_dim))

    def forward(self, x):
        for layer in self.network:
            x = layer(x)
        x_prob = torch.softmax(x.reshape(-1, 2), dim=1)
        # print(x_prob[0])
        return x_prob

class GNNModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_model, num_node, mid_layer):
        super(GNNModel, self).__init__()
        self.num_node = num_node
        self.d_model = d_model

        self.conv1 = GCNConv(in_channels, d_model)
        self.conv2 = GCNConv(d_model, d_model)

        self.network = nn.ModuleList()
        cat_dim = in_channels*num_node + num_node**2
        self.network.append(nn.Linear(self.d_model + cat_dim, self.d_model))
        self.network.append(nn.ReLU())
        self.network.append(nn.BatchNorm1d(self.d_model))
        for i in range(mid_layer):
            self.network.append(nn.Linear(self.d_model, self.d_model))
            self.network.append(nn.ReLU())
            self.network.append(nn.BatchNorm1d(self.d_model))
        self.network.append(nn.Linear(d_model, out_channels))

    def forward(self, x, edge_index, edge_weight, batch):
        x_in, edge_weight_in = x, edge_weight
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = global_mean_pool(x, batch)  # Global pooling over nodes for each graph 
        x = F.relu(x)
        batchsize = len(batch.unique())
        x = torch.cat((x, x_in.reshape(batchsize, -1), edge_weight_in.reshape(batchsize, -1)), dim=1)
        x = F.relu(x)
        for layer in self.network:
            x = layer(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, out_dim, d_model=512, nhead=4, num_encoder_layers=5, dim_feedforward=512, dropout=0.1, max_len=512):
        super(TransformerRegressor, self).__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc_out = nn.Linear(d_model, out_dim)

    def forward(self, src):
        src = self.embedding(src)
        # src = self.pos_encoder(src) # loss explode when adding position embedding
        output = self.transformer_encoder(src)
        output = output.mean(dim=0)  # Global average pooling
        output = self.fc_out(output)
        return output

