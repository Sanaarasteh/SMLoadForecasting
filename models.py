import torch

import torch.nn as nn
import numpy as np

from source.layers import GraphIsomorphismNetwork, GraphLSTM


class GNNLSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, lstm_hidden_size,
                 lstm_num_layers, batch_size, gnn_hidden_size, lstm_dropout=0):
        super(GNNLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.gnn_hidden_size = gnn_hidden_size
        self.lstm_dropout = lstm_dropout
        self.output_dim = output_dim
        self.batch_size = batch_size

        # LSTM layers definition
        self.graph_lstm = GraphLSTM(num_nodes, input_dim, lstm_hidden_size, lstm_num_layers, batch_size, lstm_dropout)

        # GNN layers definition

        self.gin = GraphIsomorphismNetwork(lstm_num_layers * lstm_hidden_size, gnn_hidden_size)

        # MLP definition
        self.mlp = nn.Sequential(
            nn.Linear(gnn_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        lstm_outputs, lstm_hidden_states = self.graph_lstm(x)

        lstm_feature_vectors = []

        for i in range(self.num_nodes):
            size = lstm_hidden_states[i][0].size()
            lstm_feature_vectors.append(lstm_hidden_states[i][0].view(size[1], size[0] * size[2]))

        adj = np.cov(lstm_feature_vectors)
        adj[adj >= 0.5] = 1.
        adj[adj < 0.5] = 0.

        out = self.gin(lstm_feature_vectors, torch.tensor(adj))

        return out
