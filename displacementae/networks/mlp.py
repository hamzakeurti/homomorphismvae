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
# @title          :displacementae/networks/mlp.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Simple MLP network

    Network made of stacked linear layers, optionally activated.
    """

    def __init__(self, in_features, out_features, hidden_units,
                 activation=nn.ReLU, dropout_rate=0, bias=True, layer_norm=False):
        super().__init__()
        layers = []

        units = [in_features] + hidden_units
        for l in range(len(units)-1):
            n_in = units[l]
            n_out = units[l + 1]
            layers.append(
                nn.Linear(in_features=n_in, out_features=n_out, bias=bias))
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            if layer_norm:
                layers.append(nn.LayerNorm(n_out))
            layers.append(activation())

        layers.append(nn.Linear(n_out, out_features))

        self.seq = nn.Sequential(*layers)

    def forward(self, x):
        """Computes the output of this network.

        Args:
            x (tensor):

        Returns:
            tensor: output tensor, not activated.
        """

        return self.seq(x)
