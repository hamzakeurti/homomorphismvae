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

import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from models import nets

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

class OrthogonalMatrix(nn.Module):
    """
    A torch module that parametrizes a O(n) subgroup, to allow for matrix optimization within the orthogonal group.

    Torch module that parametrizes a group of orthogonal matrices. 
    The module does not have any weights, it just takes a vector of parameters and turns them into an orthogonal matrix.
    """
    CAYLEY = 'cayley'
    BLOCKS = 'blocks'

    def __init__(self, transformation=CAYLEY, n_units=6,device = 'cpu', 
                learn_params=False):
        nn.Module.__init__(self)
        self.device = device
        self.transformation = transformation
        self.n_units = n_units        
        self.matrix_size = n_units
        if transformation == OrthogonalMatrix.CAYLEY:
            self.n_parameters = self.n_units*(self.n_units-1)/2
        elif transformation == OrthogonalMatrix.BLOCKS:
            if self.n_units % 2 == 1:
                raise ValueError(
                    'Latent space should have an even dimension for the matrix to be expressed in blocks of 2.')
            self.n_parameters = self.n_units//2  # Number of blocks
            self.rot_basis = torch.FloatTensor([
                [[1., 0.],
                 [0., 1.]],
                [[0., -1.],
                 [1., 0.]]]).to(device)
        self.learn_params = learn_params
        if learn_params:
            self.alpha = nn.parameter.Parameter(
                torch.rand([self.n_parameters])/10)

    def forward(self, parameters):
        if parameters.shape[-1] != self.n_parameters:
            raise ValueError(
                f'Expected input last dimension to be {self.n_parameters}, received {parameters.shape[-1]}')
        if self.transformation == OrthogonalMatrix.BLOCKS:
            # parameters shape: [b,n_parameters]
            return _block_diag(self.rotation_matrices(parameters))

    def rotation_matrices(self, angle):
        u = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        return torch.tensordot(u, self.rot_basis, dims=[[-1], [0]])
    
    def transform(self, h, angles):
        """
        Function that populates an orthogonal matrix, 
        then rotates the input vector h through the obtained matrix.
        
        Args:
            h: vector to transform
            angles: parameters of the rotation matrix
        """
        if self.learn_params:
            angles = self.alpha*angles
        O = self.forward(angles)
        return torch.matmul(O, h.unsqueeze(-1)).squeeze(dim=-1)