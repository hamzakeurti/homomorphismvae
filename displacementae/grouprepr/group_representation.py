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
from grouprepr.varphi import VarPhi


class GroupRepresentation(nn.Module):
    """
    An interface for group representation classes.

    Group representations are maps from the abstract action group to
    invertible matrices through the :method:`forward` method.
    Group representations also define a linear
    action of the abstract group on the representation space of
    observations, as such the group representation transforms input
    representation vectors through the matrix product with the
    representation of a given action, through the :method:`act` method.
    """
    def __init__(self, n_action_units:int, dim_representation:int, device='cpu',
                 repr_loss_on = False, repr_loss_weight = 0., 
                 normalize= False, normalize_post_action=False, 
                 varphi_units:list=[],
                 varphi_seed:int=1) -> None:
        super().__init__()
        self.device = device
        self.n_action_units = n_action_units
        self.varphi_units = varphi_units
        self.dim_representation = dim_representation
        self.repr_loss_on = repr_loss_on
        self.repr_loss_weight = repr_loss_weight
        self.normalize = normalize
        self.normalize_post_action = normalize_post_action
        

        self.varphi = VarPhi(n_action_units=n_action_units,
                             device=device, linear_units=varphi_units,
                             seed=varphi_seed).to(device)
        self.varphi_out = self.varphi.out_units

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Gets the representation matrix of input transition :arg:`a`.
        """
        pass

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Acts on an input representation vector :arg:`z` through matrix
        product with the representation matrix of input transition
        :arg:`a`.

        Args:
            a, torch.Tensor: Batch of transitions.
                        shape: `[batch_size,n_action]`
            z, torch.Tensor: Batch of representation vectors.
                        shape: `[batch_size,n_repr]`
        Returns:
            torch.Tensor: Transformed representation vectors.
                        shape: `[batch_size,n_repr_units]`
        """
        z_out =  torch.einsum("...jk,...k->...j",self.forward(a),z)
        if self.normalize_post_action:
            z_out = self.normalize_vector(z_out)

        return z_out

    def get_example_repr(self, a: torch.Tensor = None) -> np.ndarray:
        with torch.no_grad():
            if a is None:
                a = torch.zeros(self.n_action_units*2+1,
                                self.n_action_units,
                                device=self.device)
                for i in range(self.n_action_units):
                    a[1+2*i:3+2*i,i] = torch.tensor([1,-1])

            R = self.forward(a)

            if R.device.type == 'cuda':
                R = R.cpu()
            return R.numpy()

    def representation_loss(self, *args):
        pass

    def end_iteration(self):
        pass

    def normalize_vector(self, z: torch.Tensor):
        if self.normalize:
            return F.normalize(z, dim=-1)
        else:
            return z
