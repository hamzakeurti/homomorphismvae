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
# @title          :displacementae/networks/variational_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :18/11/2021
# @version        :1.0
# @python_version :3.7.4

import torch

def reparametrize(mu, logvar):
    """
    Reparametrization trick for sampling from a gaussian of 
    which parameters are outputted by a model.
    """
    eps = torch.randn_like(mu)
    std = torch.exp(0.5*logvar)
    return mu + std*eps

def kl_loss(mu, logvar):
    loss = torch.exp(logvar) + mu**2 - 1. - logvar
    return 0.5 * torch.sum(loss)