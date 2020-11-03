# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

class UpsdBehavior(nn.Module):
    '''
    Behavour function that produces actions based on a state and command.
    Schmidhuber. 
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (int) -- NOTE: not used at the moment
        desires_scalings (List of float)
    '''
    
    def __init__(self, state_size, desires_size,
            action_size, hidden_sizes,
            desires_scalings ):
        super().__init__()
        self.desires_scalings = torch.FloatTensor(desires_scalings[:desires_size])
        
        self.state_fc = nn.Sequential(nn.Linear(state_size, hidden_sizes[0]), 
                                      nn.Tanh())
        
        self.command_fc = nn.Sequential(nn.Linear(desires_size, hidden_sizes[0]), 
                                        nn.Sigmoid())

        layers = nn.ModuleList()
        hidden_sizes.append(action_size)
        output_activation= nn.Identity
        activation = nn.ReLU
        for j in range(len(hidden_sizes)-1):
            act = activation if j < len(hidden_sizes)-2 else output_activation
            layers.append(nn.Linear(hidden_sizes[j], hidden_sizes[j+1]) )
            layers.append(act())
        self.output_fc = nn.Sequential(*layers)
    
    def forward(self, state, command):
        '''Forward pass
        
        Params:
            state (List of float)
            command (List of float)
        
        Returns:
            FloatTensor -- action logits
        '''
        if len(command)==1:
            command=command[0]
        else: 
            command = torch.cat(command, dim=1)
        #print('entering the model', state.shape, command.shape)
        state_output = self.state_fc(state)
        command_output = self.command_fc(command * self.desires_scalings)
        embedding = torch.mul(state_output, command_output)
        return self.output_fc(embedding)

class UpsdModel(nn.Module):
    #TODO: get the desire state for this working. 
    """ Using Fig.1 from Reward Conditioned Policies 
        https://arxiv.org/pdf/1912.13465.pdf """
    def __init__(self, state_size, desires_size, 
        action_size, hidden_sizes,
        state_act_fn="relu", desires_act_fn="sigmoid"):
        super().__init__()
        self.state_act_fn = getattr(torch, state_act_fn)
        self.desires_act_fn = getattr(torch, desires_act_fn)

        self.state_layers = nn.ModuleList()
        self.desires_layers = nn.ModuleList()
        hidden_sizes.insert(0, state_size)
        for j in range(len(hidden_sizes)-1):
            self.state_layers.append( nn.Linear(hidden_sizes[j], hidden_sizes[j+1]) )
            self.desires_layers.append( nn.Linear(desires_size, hidden_sizes[j+1]) )

        self.output_fc = nn.Linear(hidden_sizes[-1], action_size )

    def forward(self, state, desires):
        ''' Returns an action '''

        # always will want the first element of the list. 
        if len(desires)==1:
            desires=desires[0]
        else: 
            desires = torch.cat(desires, dim=1)

        for state_layer, desires_layer in zip(self.state_layers, self.desires_layers):
            state = torch.mul( self.state_act_fn(state_layer(state)), 
                    self.desires_act_fn(desires_layer(desires))   )

        state = self.output_fc(state)

        return state