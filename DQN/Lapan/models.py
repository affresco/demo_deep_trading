import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class SimpleFFDQN(nn.Module):
    def __init__(self, obs_len, actions_n):
        super(SimpleFFDQN, self).__init__()

        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(obs_len, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, actions_n)
        )

    def forward(self, x):
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class DQNConv1D(nn.Module):

    def __init__(self, shape, actions_n):
        super(DQNConv1D, self).__init__()

        self.conv = nn.Sequential(

            nn.Conv1d(shape[0], 512, 5),
            nn.MaxPool1d(3, 2),
            nn.LeakyReLU(),

            nn.Conv1d(512, 256, 3),
            nn.MaxPool1d(3, 2),
            nn.LeakyReLU(),

            nn.Conv1d(256, 128, 3),
            nn.MaxPool1d(3, 2),
            nn.LeakyReLU(),

            nn.Conv1d(128, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Flatten(),

        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(

            nn.Linear(out_size, 1024),
            nn.LeakyReLU(0.001),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.001),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.001),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.001),

            nn.Linear(64, 1)
        )

        self.fc_adv = nn.Sequential(

            nn.Linear(out_size, 1024),
            nn.LeakyReLU(0.001),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.001),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.001),

            nn.Linear(128, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class DQNConv1D_plus_GRU(nn.Module):

    def __init__(self, shape, actions_n):

        super(DQNConv1D_plus_GRU, self).__init__()

        self.input_shape = shape

        self.conv = nn.Sequential(

            nn.Conv1d(shape[0], 512, 5),
            nn.AvgPool1d(3, 1),
            nn.LeakyReLU(),

            nn.Conv1d(512, 256, 5),
            nn.AvgPool1d(3, 1),
            nn.LeakyReLU(),

            nn.Conv1d(256, 128, 3),
            nn.AvgPool1d(3, 1),
            nn.LeakyReLU(),

            nn.Conv1d(128, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Conv1d(64, 64, 3),
            nn.LeakyReLU(),

            nn.Flatten(),

        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(

            nn.Linear(out_size, 1024),
            nn.LeakyReLU(0.001),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.001),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.001),

            nn.Linear(128, 64),
            nn.LeakyReLU(0.001),

            nn.Linear(64, 1)
        )

        self.fc_adv = nn.Sequential(

            nn.Linear(out_size, 1024),
            nn.LeakyReLU(0.001),

            nn.Linear(1024, 512),
            nn.LeakyReLU(0.001),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 256),
            nn.LeakyReLU(0.001),

            nn.Linear(256, 128),
            nn.LeakyReLU(0.001),

            nn.Linear(128, actions_n)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.c1 = nn.Conv1d(input_size, hidden_size, 2)
        self.p1 = nn.AvgPool1d(2)
        self.c2 = nn.Conv1d(hidden_size, hidden_size, 1)
        self.p2 = nn.AvgPool1d(2)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=0.01)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, inputs, hidden):
        batch_size = inputs.size(1)

        # Turn (seq_len x batch_size x input_size) into (batch_size x input_size x seq_len) for CNN
        inputs = inputs.transpose(0, 1).transpose(1, 2)

        # Run through Conv1d and Pool1d layers
        c = self.c1(inputs)
        p = self.p1(c)
        c = self.c2(p)
        p = self.p2(c)

        # Turn (batch_size x hidden_size x seq_len) back into (seq_len x batch_size x hidden_size) for RNN
        p = p.transpose(1, 2).transpose(0, 1)

        p = F.tanh(p)
        output, hidden = self.gru(p, hidden)
        conv_seq_len = output.size(0)
        output = output.view(conv_seq_len * batch_size,
                             self.hidden_size)  # Treating (conv_seq_len x batch_size) as batch_size for linear layer
        output = F.tanh(self.out(output))
        output = output.view(conv_seq_len, -1, self.output_size)
        return output, hidden
