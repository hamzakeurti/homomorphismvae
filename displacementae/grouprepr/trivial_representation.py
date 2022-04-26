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
# @title          :displacementae/grouprepr/trivial_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :24/03/2022
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn

from grouprepr.group_representation import GroupRepresentation

class TrivialRepresentation(GroupRepresentation):
    """
    A Trivial Representation. 
    
    Maps all group elements to identity transformation.
    """
    def __init__(self, dim_representation: int,
                 device='cpu') -> None:
        super().__init__(n_action_units=1, 
                         dim_representation=dim_representation, device=device)
    
    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Returns identity representation for any action.
        """
        d = self.dim_representation
        n = a.shape[:-1]
        return torch.eye(d).repeat([*n,1,1])

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Acts trivially on input z.
        """
        return z

if __name__ == '__main__':
    pass
    
