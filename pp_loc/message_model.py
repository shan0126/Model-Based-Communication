import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MessageModel(nn.Module):
    def __init__(self, args):
        super(MessageModel, self).__init__()
        """
        Initialization method for the generator
        """

        self.message_size = args.hid_size
        self.nagents = args.nagents
        self.nactions = args.nactions
        self.hidden = 500

        self.fc1 = nn.Linear((self.message_size+5*self.nagents) * self.nagents, self.hidden)  # input without action (1 hot)

        self.fc2 = nn.Linear(self.hidden, self.message_size * self.nagents)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialization for the parameters of the graph generator
        """
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.fc1.weight.data, gain=gain)
        # nn.init.xavier_normal_(self.fc1.bias.data, gain=gain)
        nn.init.xavier_normal_(self.fc2.weight.data, gain=gain)
        # nn.init.xavier_normal_(self.fc2.bias.data, gain=gain)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x

