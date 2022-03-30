#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti
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
# @title          :networks/geometric/prodrepr/product_represenation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :04/02/2022
# @version        :1.0
# @python_version :3.7.4
"""
Group representation of actions into a product of rotation matrices
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adapted from 
`Quessard et al. 2020 <https://github.com/IndustAI/learning-group-structure>`.
"""

import os
import torch.nn as nn
import torch
import numpy as np

from grouprepr.prodrepr.product_representation import ProductRepresentation
from grouprepr.group_representation import GroupRepresentation

class ActionLookup(GroupRepresentation):
    def __init__(self, n_action_units: int, dim_representation: int, 
                 repr_loss_on=False, repr_loss_weight=0, device='cpu') -> None:
        super().__init__(n_action_units, dim_representation, 
                         repr_loss_on, repr_loss_weight)
        self.device = device
        self.action_reps = nn.ModuleList([
            ProductRepresentation(dim_representation,device) 
            for _ in range(n_action_units)])
        
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        a = a.int()
        R = [self.action_reps[a[i]].get_matrix() for i in range(a.shape[0])]
        R = torch.stack(R,dim=0)
        return R

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return super().act(a, z)

    def get_representation_params(self):
        params = []
        for rep in self.action_reps:
            params.append(rep.thetas)
        return params
    
    def end_iteration(self):
        self.clear_representations() 

    def representation_loss(self, *args):
        return self.entanglement_loss()

    def clear_representations(self):
        for rep in self.action_reps:
            rep.clear_matrix()
    
    def save_representations(self, path):
        if os.path.splitext(path)[-1] != '.pth':
            path += '.pth'
        rep_thetas = [rep.thetas for rep in self.action_reps]
        return torch.save(rep_thetas, path)
    
    def load_reprentations(self, path):
        rep_thetas = torch.load(path)
        for rep in self.action_reps:
            rep.set_thetas(rep_thetas.pop(0))

    def entanglement_loss(self):
        loss = 0
        for rep in self.action_reps:
            loss += rep.entanglement_loss()
        return loss/len(self.action_reps)

    def get_example_repr(self, a: torch.Tensor = None) -> np.ndarray:
        with torch.no_grad():
            if a is None:
                a = torch.IntTensor([0,1])
            return super().get_example_repr(a)