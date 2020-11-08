# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

class AdvantageModel(nn.Module):
    '''
    This is the model from AWR: 
    https://github.com/xbpeng/awr/blob/1e7f8130476f138952de18d3d68e8e6c45d36d6a/learning/awr_agent.py
    And uses some of its scripts. 
    '''
    
    def __init__(self, state_size, hidden_sizes=[128, 64], 
            act_fn="ReLU"):
        super().__init__()
        self.act_fn = getattr(torch.nn, act_fn)
        hidden_sizes.insert(0, state_size)
        self.layers = nn.ModuleList()
        for j in range(len(hidden_sizes)-1):
            self.layers.append( nn.Linear(hidden_sizes[j], hidden_sizes[j+1])  )
            self.layers.append( self.act_fn() )
        self.layers = nn.Sequential(*self.layers)
        self.output_fc = nn.Linear(hidden_sizes[-1], 1 )

    def forward(self, state):
        v = self.layers(state)
        v = self.output_fc(v)
        return v

    def calculate_lambda_target(self, vals, rewards, discount, td_lambda):
        # compute the advantages to use as desires during training. 
        # I then do MSE loss between the these advantages and the value function
        #vals = self.forward(states).squeeze()
        # appending zero to the end. 
        # TODO: if terminal set to 0, otherwise use the value that was given. 
        # I always run rollouts till they terminate currently so this is not a problem. 
        
        vals = torch.cat( [vals, torch.Tensor([0])], axis=0)
        td_lambda_target = self._compute_return(vals, rewards, discount, td_lambda)
        return torch.as_tensor(td_lambda_target, dtype=torch.float32)

    def _compute_return(self, val_t, rewards, discount, td_lambda):
        # computes td-lambda return of path
        # TODO: vectorize all of this or find a vectorized version. 
        path_len = len(rewards)
        assert len(val_t) == path_len + 1
        return_t = np.zeros(path_len)
        last_val = rewards[-1] + discount * val_t[-1]
        return_t[-1] = last_val
        for i in reversed(range(0, path_len - 1)):
            curr_r = rewards[i]
            next_ret = return_t[i + 1]
            curr_val = curr_r + discount * ((1.0 - td_lambda) * val_t[i + 1] + td_lambda * next_ret)
            return_t[i] = curr_val
    
        return return_t





