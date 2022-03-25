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
# @title          :displacementae/grouprepr/block_mlp_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :24/03/2022
# @version        :1.0
# @python_version :3.7.4

from typing import Callable
import numpy as np
import torch
import torch.nn as nn

from grouprepr.group_representation import GroupRepresentation
from grouprepr.mlp_representation import MLPRepresentation

class BlockMLPRepresentation(GroupRepresentation):
    """
    This subclass of group representations is a direct product of 
    subrepresentations, as such it maps transitions to block diagonal 
    matrices.

    """
    def __init__(self, n_action_units:int, dim_representation:int, 
                 dims:list, hidden_units:list=[],
                 activation:Callable=torch.relu, device:str='cpu') -> None:
        super().__init__(n_action_units, dim_representation)
        self.dims = dims
        self.n_subreps = len(dims)
        self.cumdims = [0, *np.cumsum(self.dims)]
        self.subreps:nn.ModuleList[GroupRepresentation] = nn.ModuleList()
        for dim in dims:
            self.subreps.append(
                    MLPRepresentation(n_action_units,dim,
                                      hidden_units=hidden_units,
                                      activation=activation, device=device))
            
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        R = torch.zeros(*a.shape[:-1],self.dim_representation,self.dim_representation)
        for i in range(self.n_subreps):
            R[...,self.cumdims[i]:self.cumdims[i+1], 
              self.cumdims[i]:self.cumdims[i+1]] = self.subreps[i](a)
        return R

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_out = torch.empty_like(z)
        for i in range(self.n_subreps):
            z_out[...,self.cumdims[i]:self.cumdims[i+1]] =\
                    self.subreps[i].act(
                        a,z[...,self.cumdims[i]:self.cumdims[i+1]])
        return z_out
    

if __name__ == '__main__':
    pass