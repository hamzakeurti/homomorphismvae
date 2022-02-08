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

import networks.geometric.prodrepr.product_representation as rep

class ActionLookup(nn.Module):
    def __init__(self, n_actions, dim):
        super().__init__()
        self.dim = dim
        self.n_actions = n_actions
        self.action_reps = [
            rep.ProductRepresentation(dim) for _ in range(n_actions)]
        
    def forward(self, z, action):
        """
        Performs the transformation of z by rep of action.

        Args:
            z (): input states representation vectors.
            action (): input action index.

        ReturnsL
            (): R@z, where R is the representation of the input action.
        """
        
        R = self.action_reps[action].get_matrix()
        return R@z


    def get_representation_params(self):
        params = []
        for rep in self.action_reps:
            params.append(rep.thetas)
        return params
    
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
