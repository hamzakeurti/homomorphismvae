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
                 activation = torch.nn.ReLU(), device: str = 'cpu',
                 normalize_subrepresentations = False,
                 normalize_post_action:bool=False,
                 exponential_map:bool=False,
                 varphi_units:list=[],
                 varphi_seed:int=0) -> None:
        super().__init__(n_action_units, dim_representation, device=device, 
                         normalize_post_action=normalize_post_action,
                         varphi_units=varphi_units,
                         varphi_seed=varphi_seed)
        self.exponential_map = exponential_map
        self.dims = dims
        self.n_subreps = len(dims)
        self.cumdims = [0, *np.cumsum(self.dims)]
        self.subreps: nn.ModuleList[GroupRepresentation] = nn.ModuleList()
        for dim in dims:
            self.subreps.append(
                    MLPRepresentation(
                            self.varphi_out,
                            dim,
                            hidden_units=hidden_units,
                            activation=activation, 
                            device=device, 
                            layer_norm=True,
                            normalize=normalize_subrepresentations, 
                            normalize_post_action=normalize_post_action,
                            exponential_map=exponential_map))
            
    def forward(self, a: torch.Tensor, use_exponential:bool=None) -> torch.Tensor:
        if use_exponential is None:
            use_exponential = self.exponential_map
        a = self.varphi(a)
        d = self.dim_representation
        R = torch.zeros(*a.shape[:-1],d,d,device=a.device)
        for i in range(self.n_subreps):
            R[..., self.cumdims[i]:self.cumdims[i+1],
              self.cumdims[i]:self.cumdims[i+1]] = self.subreps[i](a,use_exponential=use_exponential)
        return R

    def forward_algebra(self,a: torch.Tensor) -> torch.Tensor:
        """
        Forwards input transitions through an MLP network and reshapes
        the outputs to form matrices. No exponential.
        """
        a = self.varphi(a)
        d = self.dim_representation
        R = torch.zeros(*a.shape[:-1],d,d,device=a.device)
        for i in range(self.n_subreps):
            R[..., self.cumdims[i]:self.cumdims[i+1],
              self.cumdims[i]:self.cumdims[i+1]] = self.subreps[i].forward_algebra(a)
        return R

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        a = self.varphi(a)
        z_out = torch.zeros_like(z, device=a.device)
        for i in range(self.n_subreps):
            z_out[..., self.cumdims[i]:self.cumdims[i+1]] =\
                    self.subreps[i].act(
                        a, z[..., self.cumdims[i]:self.cumdims[i+1]])
        return z_out

    def normalize_vector(self, z: torch.Tensor):
        z_out = z
        for i in range(self.n_subreps):
            z_out[...,self.cumdims[i]:self.cumdims[i+1]] =\
                    self.subreps[i].normalize_vector(
                        z[...,self.cumdims[i]:self.cumdims[i+1]].clone())   
        return z_out

if __name__ == '__main__':
    pass
