import torch

import torch.nn as nn


class GraphIsomorphismNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphIsomorphismNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.mlp1 = nn.Sequential(
            nn.Linear(input_dim, 2 * input_dim),
            nn.ReLU(),
            nn.Linear(2 * input_dim, 2 * input_dim),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(2 * input_dim, 3 * input_dim),
            nn.ReLU(),
            nn.Linear(3 * input_dim, 2 * input_dim),
            nn.ReLU()
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(2 * input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, adj, x):
        adj_tilde = adj + torch.eye(adj.size(0))

        out = torch.matmul(adj_tilde, x)
        out = self.mlp1(out)

        out = torch.matmul(adj_tilde, out)
        out = self.mlp2(out)

        out = torch.matmul(adj_tilde, out)
        out = self.mlp3(out)

        return out


class GraphLSTM(nn.Module):
    def __init__(self, num_nodes, input_dim, hidden_size, num_layers, batch_size, dropout=0):
        super(GraphLSTM, self).__init__()

        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.dropout = dropout

        self.lstm_modules = nn.ModuleList()

        for i in range(self.num_nodes):
            lstm = nn.LSTM(self.input_dim, self.hidden_size, self.num_layers,
                           batch_first=True, dropout=self.dropout)
            self.lstm_modules.append(lstm)

        # initialize hidden and cell states of the LSTM modules
        self.hidden_states = []

        for i in range(self.num_nodes):
            self.hidden_states.append((torch.rand(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size),
                                       torch.rand(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size)))

    def forward(self, x):
        outputs = torch.rand(self.num_nodes)

        for i in range(self.num_nodes):
            output, self.hidden_states[i] = self.lstm_modules[i](x[:, i, :, :])
            outputs[i] = output

        return outputs, self.hidden_states
