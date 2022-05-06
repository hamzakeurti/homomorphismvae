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
# @title          :displacementae/grouprepr/blockrots/orthogonal.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4

import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from models import nets

from grouprepr.group_representation import GroupRepresentation


def _block_diag(m):
    """
    Taken from `here <https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168>`__
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)
    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    device = m.device
    eye = _attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1).to(device)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def _attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))

class OrthogonalMatrix(GroupRepresentation):
    """
    A torch module that parametrizes a O(n) subgroup, to allow for matrix optimization within the orthogonal group.

    Torch module that parametrizes a group of orthogonal matrices. 
    The module does not have any weights, it just takes a vector of parameters and turns them into an orthogonal matrix.
    """

    def __init__(self, dim_representation: int,
                 device:str = 'cpu', learn_params:bool=False, 
                 specified_step=0) -> None:
        super().__init__(dim_representation//2, dim_representation,
                         device=device)

        if self.dim_representation % 2 == 1:
            raise ValueError(
                'Latent space should have an even dimension for the matrix to be expressed in blocks of 2.')

        self.specified_step = specified_step

        self.rot_basis = torch.FloatTensor([
                    [[1., 0.],
                    [0., 1.]],
                    [[0., -1.],
                    [1., 0.]]]).to(device)

        self.learn_params = learn_params
        if learn_params:
            self.alpha = nn.parameter.Parameter(
                torch.rand([self.n_action_units])/10)

    def forward(self, a):
        if a.shape[-1] != self.n_action_units:
            raise ValueError(
                f'Expected input last dimension to be {self.n_action_units}, received {a.shape[-1]}')

        return _block_diag(self.rotation_matrices(a))

    def rotation_matrices(self, angle):
        u = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        return torch.tensordot(u, self.rot_basis, dims=[[-1], [0]])
    
    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Function that populates an orthogonal matrix, 
        then rotates the input vector h through the obtained matrix.
        
        Args:
            a: parameters of the rotation matrix
            z: vector to transform
        """
        if self.learn_params:
            a = self.alpha * a
        else:
            a = self.specified_step * a
        O = self.forward(a)
        return torch.matmul(O, z.unsqueeze(-1)).squeeze(dim=-1)

    def get_example_repr(self, a: torch.Tensor = None) -> np.ndarray:
        with torch.no_grad():            
            if a is None:
                alpha = self.alpha.clone()
                if alpha.device.type == 'cuda':
                    alpha = alpha.cpu()
                alpha = alpha.numpy()
                return alpha 
            else:
                return super().get_example_repr(a)
        
    def representation_loss(self, *args):
        return 0