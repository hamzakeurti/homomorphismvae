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
    def __init__(self, n_action_units:int, n_repr_units:int) -> None:
        super().__init__()
        self.n_action_units = n_action_units
        self.n_repr_units = n_repr_units

    def forward(self,a:torch.Tensor) -> torch.Tensor:
        """
        Gets the representation matrix of input transition :arg:`a`.
        """
        pass

    def act(self, a:torch.Tensor, z:torch.Tensor) -> torch.Tensor:
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
        return torch.einsum("...jk,...k->...j",self.forward(a),z)