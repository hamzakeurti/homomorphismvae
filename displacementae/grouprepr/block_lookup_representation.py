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
# @title          :displacementae/grouprepr/block_lookup_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :05/04/2022
# @version        :1.0
# @python_version :3.7.4

from typing import Callable
import numpy as np
import torch
import torch.nn as nn

from displacementae.grouprepr.group_representation import GroupRepresentation
from displacementae.grouprepr.lookup_representation import LookupRepresentation
from displacementae.grouprepr.varphi import VarPhi

class BlockLookupRepresentation(GroupRepresentation):
    """
    This subclass of group representations is a direct product of
    subrepresentations, as such it maps transitions to block diagonal
    matrices.

    """
    def __init__(self,
                 n_actions: int,
                 dim_representation: int,
                 dims: list,
                 device: str = 'cpu',
                 normalize_subrepresentations: bool = False,
                 normalize_post_action: bool = False,
                 exponential_map: bool = False,
                 varphi: VarPhi = None,
                 ) -> None:
        super().__init__(n_action_units=1,
                         dim_representation=dim_representation, device=device,
                         normalize_post_action=normalize_post_action,
                         varphi=varphi)
        self.dims = dims
        self.n_actions = n_actions
        self.n_subreps = len(dims)
        self.cumdims = [0, *np.cumsum(self.dims)]
        self.subreps: list[LookupRepresentation] = nn.ModuleList()
        for dim in dims:
            self.subreps.append(
                    LookupRepresentation(
                            n_actions,
                            dim,
                            device=device,
                            normalize=normalize_subrepresentations,
                            normalize_post_action=normalize_post_action,
                            exponential_map=exponential_map,
                            ))

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = self.varphi(a)
        R = torch.zeros(*a.shape, self.dim_representation,
                        self.dim_representation, device=a.device)
        for i in range(self.n_subreps):
            R[...,self.cumdims[i]:self.cumdims[i+1],
              self.cumdims[i]:self.cumdims[i+1]] = self.subreps[i](a)
        return R

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z_out = torch.empty_like(z,device=a.device)
        for i in range(self.n_subreps):
            z_out[...,self.cumdims[i]:self.cumdims[i+1]] =\
                    self.subreps[i].act(
                        a,z[...,self.cumdims[i]:self.cumdims[i+1]])
        return z_out

    def normalize_vector(self, z: torch.Tensor):
        z_out = z
        for i in range(self.n_subreps):
            z_out[...,self.cumdims[i]:self.cumdims[i+1]] =\
                    self.subreps[i].normalize_vector(
                        z[...,self.cumdims[i]:self.cumdims[i+1]].clone())
        return z_out

if __name__ == '__main__':
    n_action_units = 5
    n_repr_units = 5
    dims = [2,1,2]
    repr = BlockLookupRepresentation(n_action_units,n_repr_units,dims)

    batch_size = 20
    a = torch.randint(n_action_units,size=(batch_size,))
    R = repr(a)
    print(R.shape)
