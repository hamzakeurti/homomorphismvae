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
# @title          :displacementae/networks/geometric/homomorphism.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :27/01/2022
# @version        :1.0
# @python_version :3.7.4
"""
Homomorphism as a network: takes group actions as input and outputs a 
transformation of the representation space.
"""

import torch
import torch.nn as nn
import networks.mlp as mlp

class NeuralHomomorphism(nn.Module):
    def __init__(self, n_actions, n_out,layers = [20,20,20], mode='matrix'):
        self.mode = mode
        self.n_actions = n_actions
        self.n_out = n_out
        self.net = mlp.MLP(n_actions, n_out, layers)
        # TODO: Figure out output shape

    def forward(self,action):
        out = self.net(action)
        # TODO: This needs some postprocessing
        return out
