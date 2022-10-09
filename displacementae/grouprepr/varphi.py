#!/usr/bin/env python3
# Copyright 2022 Hamza Keurti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :displacementae/grouprepr/group_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :21/03/2022
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from networks.mlp import MLP


class VarPhi(nn.Module):
    """
    A mixing function of the action units.

    """
    def __init__(self, n_action_units:int, device='cpu',
                 linear_units:list=[], seed:int=1) -> None:
        super().__init__()
        self.device = device
        self.n_action_units = n_action_units
        self.linear_units = linear_units
        
        if self.linear_units==[0]:
            self.linear_units=[]
        if self.linear_units==[]:
            self.net = nn.Identity()
            self.out_units = self.n_action_units
        else:
            self.out_units = self.linear_units[-1]
            self.net = MLP(in_features=n_action_units,
                              out_features=self.out_units,
                              hidden_units=self.linear_units[:-1],
                              activation=nn.LeakyReLU(negative_slope=0.2),seed=seed,bias=False)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.eval()
            return self.net(a) 
    

