# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F

class RewardModel(nn.Module):
    def __init__(self, hidden_size, state_size, node_size, act_fn="relu"):
        super().__init__()
        self.act_fn = getattr(torch, act_fn)
        self.fc_1 = nn.Linear(hidden_size + state_size, node_size)
        self.fc_2 = nn.Linear(node_size, node_size)
        self.fc_3 = nn.Linear(node_size, 1)

    def forward(self, hidden, state):
        out = self.act_fn(self.fc_1(torch.cat([hidden, state], dim=1)))
        out = self.act_fn(self.fc_2(out))
        reward = self.fc_3(out).squeeze(dim=1)
        return reward