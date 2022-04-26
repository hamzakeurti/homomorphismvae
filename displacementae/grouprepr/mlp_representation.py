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
# @title          :displacementae/grouprepr/mlp_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :24/03/2022
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn
import torch.nn.functional as F


from grouprepr.group_representation import GroupRepresentation
from networks.mlp import MLP

class MLPRepresentation(GroupRepresentation):
    """
    An MLP mapping from transitions to invertible matrices.

    """
    def __init__(self, n_action_units: int, dim_representation: int, 
                 hidden_units=[], activation=torch.relu, normalize=False, 
                 device='cpu') -> None:
        super().__init__(n_action_units, dim_representation, device=device, 
                         normalize=normalize)
        self.net = MLP(n_action_units,dim_representation**2,hidden_units,
                       activation).to(device)
                       
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Forwards input transitions through an MLP network and reshapes 
        the outputs to form matrices. 
        """
        R = self.net(a)
        R = R.view(-1, self.dim_representation,self.dim_representation)
        return R

if __name__ == '__main__':
    pass
    
