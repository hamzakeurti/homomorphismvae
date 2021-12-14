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
                activation=torch.relu, dropout_rate=0, bias=True):
        super().__init__()
        self._layers = nn.ModuleList()

        units = [in_features] + hidden_units + [out_features]
        for l in range(len(units)-1):
            n_in = units[l]
            n_out = units[l + 1]
            self._layers.append(
                nn.Linear(in_features=n_in, out_features=n_out, bias=bias))

        self._activation = activation
        self._dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Computes the output of this network.

        Args:
            x (tensor): 

        Returns:
            tensor: output tensor, not activated.
        """
        h = x
        for l in range(len(self._layers)):
            h = self._layers[l](h)
            if l != len(self._layers) - 1:
                h = self._dropout(h)
                if self._activation is not None:
                    h = self._activation(h)
        return h
