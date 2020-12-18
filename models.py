import torch
import torch.nn as nn


class GNNLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, lstm_hidden_size, lstm_num_layers, batch_size, lstm_dropout=0):
        super(GNNLSTM, self).__init__()

        self.input_dim = input_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.lstm_dropout = lstm_dropout
        self.output_dim = output_dim
        self.batch_size = batch_size

        # LSTM layers definition
        self.lstm = nn.LSTM(self.input_dim, self.lstm_hidden_size, self.lstm_num_layers,
                            batch_first=True, dropout=self.lstm_dropout)

        self.h, self.c = (torch.rand(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size),
                          torch.rand(self.lstm_num_layers, self.batch_size, self.lstm_hidden_size))

        # GNN layers definition

        # MLP definition
        self.mlp = nn.Sequential(
            nn.Linear(self.lstm_num_layers * self.lstm_hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.output_dim)
        )

    def forward(self, x):
        out, (self.h, self.c) = self.lstm(x)

        out = self.mlp(self.h.view(self.h.size(1), self.h.size(0) * self.h.size(2)))

        return out
