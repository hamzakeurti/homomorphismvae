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


from displacementae.grouprepr.group_representation import GroupRepresentation
from displacementae.grouprepr.mlp_representation import MLPRepresentation
from displacementae.networks.mlp import MLP
from displacementae.grouprepr.varphi import VarPhi


class SoftBlockMLPRepresentation(MLPRepresentation):
    """
    An MLP mapping from transitions to invertible matrices.

    """
    def __init__(self, n_action_units: int, dim_representation: int, 
                 hidden_units=[], 
                 activation=nn.ReLU(),
                 normalize=False, 
                 device='cpu',
                 layer_norm=False, 
                 normalize_post_action:bool=False,
                 exponential_map:bool=False,
                 varphi: VarPhi = None,
                 ) -> None:
        super().__init__(
                 n_action_units=n_action_units, 
                 dim_representation=dim_representation, 
                 hidden_units=hidden_units, 
                 activation=activation,
                 normalize=normalize, 
                 device=device,
                 layer_norm=layer_norm, 
                 normalize_post_action=normalize_post_action,
                 exponential_map=exponential_map,
                 varphi=varphi,
                 )
        self.masks = self._get_masks().to(device)
        self.repr_loss_on = True
        
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        return super().forward(a)

    def _get_masks(self):
        d = self.dim_representation
        M = torch.ones([d-1,d, d],dtype=bool)
        for i in range(d-1):
            M[i,:i+1,:i+1] = 0
            M[i,i+1:,i+1:] = 0
        return M

    def representation_loss(self, dj):
        R = self(dj) # Calls the underlying MLPRepresentation forward.
        l = self.masks.unsqueeze(0)*R.unsqueeze(1)
        l = l.square().sum((2,3)).sqrt().sum(1).square().sum().sqrt()
        return l
        
if __name__ == '__main__':
    pass
