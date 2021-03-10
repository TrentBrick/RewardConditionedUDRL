# pylint: disable=not-callable
# pylint: disable=no-member

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np 

class UpsdBehavior(nn.Module):
    '''
    UDRL behaviour function that produces actions based on a state and command. 
    
    Params:
        state_size (int)
        action_size (int)
        hidden_size (list of ints)
        desires_scalings (List of float)
    '''
    
    def __init__(self, state_size, desires_size,
            action_size, hidden_sizes,
            desires_scalings ):
        super().__init__()
        self.desires_scalings = torch.FloatTensor(desires_scalings)
        
        l = nn.Linear(state_size, hidden_sizes[0])
        torch.nn.init.orthogonal_(l.weight, gain=1)
        self.state_fc = nn.Sequential(l, nn.Tanh() )
        
        l = nn.Linear(desires_size, hidden_sizes[0])
        torch.nn.init.orthogonal_(l.weight, gain=1)
        self.command_fc = nn.Sequential(l, nn.Sigmoid() )                   

        layers = nn.ModuleList()
        activation = nn.ReLU
        output_activation= nn.Identity
        for j in range(len(hidden_sizes)-1):
            l = nn.Linear(hidden_sizes[j], hidden_sizes[j+1])
            torch.nn.init.orthogonal_(l.weight, gain=np.sqrt(2))
            layers.append(l)
            layers.append(activation())

        # output layer:
        # uses default Pytorch init.  
        layers.append(nn.Linear(hidden_sizes[-1], action_size ))
        layers.append(output_activation())
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


class UpsdHyper(nn.Module):
    '''
    Hypernetwork for better multiplicative conditioning. 
    Hattip to Rupesh Srivastava. 
    Reference to <https://openreview.net/forum?id=rylnK6VtDH>

    Params:
        state_size (int)
        action_size (int)
        hidden_size (list of ints)
        desires_scalings (List of float)
    '''
    
    def __init__(self, state_size, desires_size,
            action_size, hidden_sizes):
        super().__init__()

        self.activation = nn.ReLU
        self.output_activation= nn.Identity
        hidden_sizes.append(action_size)
        self.hidden_sizes = hidden_sizes
        self.embed = nn.Sequential(nn.Linear(state_size, hidden_sizes[0]), nn.Tanh() ) 

        self.hyper_weights = nn.ModuleList()
        self.hyper_biases = nn.ModuleList()
        for j in range(len(hidden_sizes)-1):
            w = nn.Linear(desires_size, hidden_sizes[j] * hidden_sizes[j+1])
            b = nn.Linear(desires_size, hidden_sizes[j+1])
            self.hyper_weights.append(w)
            self.hyper_biases.append(b)
            
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
        
        batch_size = state.shape[0]
        out = self.embed(state)

        for i in range(len(self.hidden_sizes)-1):
            w = self.hyper_weights[i](command).reshape(batch_size, self.hidden_sizes[i], self.hidden_sizes[i+1] )
            b = self.hyper_biases[i](command) 
            out = self.activation()(torch.matmul(out.unsqueeze(1), w).squeeze(1) + b)
        
        return self.output_activation()(out)

class UpsdModel(nn.Module):
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