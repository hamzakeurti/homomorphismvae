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
# @title          :displacementae/networks/autoencoder.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4

import torch.nn as nn
import torch

import networks.variational_utils as var_utils

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, grp_transformation, specified_step=0,
                 variational=True, intervene=True):
        """
        An autoencoder neural network with group transformation applied to th 

        Args:
            shape_in (tuple, optional): tuple of height, width of the input, 
                channels should be specified in conv_channels argument.
            kernel_sizes (int, optional): [description]. Defaults to 5.
            conv_channels (list, optional): [description].
            linear_channels (list, optional): [description]. Defaults to [20].
            use_bias (bool, optional): [description]. Defaults to True.
            use_bn (bool, optional): [description]. Defaults to True.
        """
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.grp_transform = grp_transformation
        self.specified_step = specified_step
        self.variational = variational
        self.intervene = intervene
        self.n_transform_units = self.grp_transform.n_units

    def forward(self, x, dz):
        h = x
        h = self.encoder(x)
        if self.variational:
            half = h.shape[1]//2
            mu, logvar = h[:, : half], h[:, half:]
            h = var_utils.reparametrize(mu, logvar)
        if self.intervene:
            # Through geom
            if not self.grp_transform.learn_params:
                dz = dz * self.specified_step
            transformed_h = \
                self.grp_transform.transform(h[:,:self.n_transform_units], dz)
            h = torch.hstack([transformed_h,h[:,self.n_transform_units:]])
        # Through decoder
        h = self.decoder(h)
        if self.variational:
            return h, mu, logvar
        else:
            return h, None, None