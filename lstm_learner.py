from __future__ import division, print_function, absolute_import

import pdb
import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np

class Learner(nn.Module):

    def __init__(self, image_size, bn_eps, bn_momentum, num_classes):
        super(Learner, self).__init__()

        self.model = nn.ModuleDict({'features': nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32, eps=bn_eps, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.2),
            nn.MaxPool2d(kernel_size=2)
        )})

        fin_size = image_size // (2*2*2*2)
        # Define the final linear layer
        self.model.update({'lin': nn.Linear(in_features=32 * fin_size * fin_size, out_features=num_classes)}) 
        
        # Define the softmax layer
        # self.model.update({'sft': nn.Softmax(dim=1)}) 
        
        # Define the cross-entropy loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, labels=None):
        # Pass the input through the CNN layers
        x = self.model.features(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Pass the output through the final linear layer and the softmax layer
        x = self.model.lin(x)
        # x = self.model.sft(x)

        return x

    def get_param_learnenr(self):
        return torch.cat([p.view(-1) for p in self.model.parameters()], 0)

    def copy_param_learnenr(self, cI):
        idx = 0
        for p in self.model.parameters():
            w = p.view(-1).size(0)
            p.data.copy_(cI[idx: idx+w].view_as(p))
            idx += w

    def transfer_params(self, learner_grad, cI):
        self.load_state_dict(learner_grad.state_dict())
        
        idx = 0
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.Linear):
                wlen = m._parameters['weight'].view(-1).size(0)
                m._parameters['weight'] = cI[idx: idx+wlen].view_as(m._parameters['weight']).clone()
                idx += wlen
                if m._parameters['bias'] is not None:
                    blen = m._parameters['bias'].view(-1).size(0)
                    m._parameters['bias'] = cI[idx: idx+blen].view_as(m._parameters['bias']).clone()
                    idx += blen

    def reset(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.reset_running_stats()

