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
# @title          :displacementae/grouprepr/lookup_representation.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :24/03/2022
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn

from grouprepr.group_representation import GroupRepresentation


class LookupRepresentation(GroupRepresentation):
    """
    A Lookup table of learned matrices.

    """
    def __init__(self,
                 n_actions: int,
                 dim_representation: int,
                 scale: float = 0.05,
                 device: str = 'cpu',
                 normalize: bool = False,
                 normalize_post_action: bool = False,
                 exponential_map: bool = False,
                 varphi_units:list=[]) -> None:
        super().__init__(n_action_units=1,
                         dim_representation=dim_representation, device=device,
                         normalize=normalize,
                         normalize_post_action=normalize_post_action,
                         varphi_units=varphi_units)
        self.action_reps = nn.ParameterList([
            nn.parameter.Parameter(
                 scale * torch.randn(size=(dim_representation, dim_representation)))
            for _ in range(n_actions)
        ])
        self.exponential_map = exponential_map

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Looks up representation matrices for each input action.
        """
        a = self.varphi(a)
        a = a.int()
        R = [self.action_reps[a[i]] for i in range(a.shape[0])]
        R = torch.stack(R, dim=0)
        if self.exponential_map:
            R = torch.matrix_exp(R)
        return R


if __name__ == '__main__':
    pass
