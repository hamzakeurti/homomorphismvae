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
Product of rotation matrices from stored angles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Adapted from 
`Quessard et al. 2020 <https://github.com/IndustAI/learning-group-structure>`.
"""

import numpy as np
import torch
import torch.nn as nn

class ProductRepresentation(nn.Module):
    def __init__(self, dim, device):
        super().__init__()
        self.device = device
        self.dim = dim
        self.n_params = int(dim*(dim-1)/2) 
        self.thetas = nn.parameter.Parameter(
            np.pi*(2*torch.rand(self.n_params)-1)/self.dim)

        self.__matrix = None
    
    def set_thetas(self, thetas):
        self.thetas = thetas
        self.thetas.requires_grad = True
        self.clear_matrix()
    
    def clear_matrix(self):
        self.__matrix = None
        
    def get_matrix(self):
        if self.__matrix is None:
            k = 0
            mats = []
            for i in range(self.dim-1):
                for j in range(self.dim-1-i):
                    theta_ij = self.thetas[k]
                    k+=1
                    c, s = torch.cos(theta_ij), torch.sin(theta_ij)

                    rotation_i = torch.eye(self.dim, self.dim,device=self.device)
                    rotation_i[i, i] = c
                    rotation_i[i, i+j+1] = s
                    rotation_i[j+i+1, i] = -s
                    rotation_i[j+i+1, j+i+1] = c

                    mats.append(rotation_i)

            def chain_mult(l):
                if len(l)>=3:
                    return l[0]@l[1]@chain_mult(l[2:])
                elif len(l)==2:
                    return l[0]@l[1]
                else:
                    return l[0]

            self.__matrix = chain_mult(mats)
                                    
        return self.__matrix

    def entanglement_loss(self):
        params = self.thetas.pow(2)
        return params.sum() - params.max()