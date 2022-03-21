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
import torch.nn.functional as F

import networks.variational_utils as var_utils

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, grp_morphism, n_repr_units, 
                 specified_step=0, variational=True, intervene=True, 
                 spherical = False):
        """
        An autoencoder neural network with group transformation applied to 
        the latent space.

        Args:
            encoder, nn.Module: Encoder Network. Maps inputs to a representation 
                            vector or to parameters of a posterior distribution 
                            in the variational case. 
            decoder, nn.Module: Decoder Network. Maps a representation 
                            vector back in the input space. 
            grp_transformation, nn.Module: Maps an action to a transformation of
                            the representation space.
            n_repr_units, int: Dimensionality of the representation space.
                            :note: In the variational case, this does not 
                            correspond to the dimensionality of the 
                            encoder's output space.
            specified_step, array: If parameters of the grp_transformation are 
                            not learned, they can be provided through this 
                            argument. defaults to 0
            variational, bool: If True, the encoder describes a distribution instead 
                            of being deterministic, defaults to True.
            intervene, bool: If true actions are provided and are used to transform 
                            the encoded representations, defaults to True.
            spherical, bool: If True, the encoder's outputs (the location part 
                            in the variational case) is normalized.
        """
        nn.Module.__init__(self)
        self.encoder = encoder
        self.decoder = decoder
        self.grp_morphism = grp_morphism
        self.specified_step = specified_step
        self.variational = variational
        self.intervene = intervene
        self.spherical = spherical
        if grp_morphism is not None:
            self.n_transform_units = self.grp_morphism.n_units
        else:
            self.n_transform_units = 0
        self.n_repr_units = n_repr_units

    def encode(self,x):
        h = self.encoder(x)
        if self.variational:
            half = h.shape[1]//2
            mu, logvar = h[:, : half], h[:, half:]
            h = var_utils.reparametrize(mu, logvar)
        if self.spherical:
            h = F.normalize(h).squeeze()
        if self.variational:
            return h, mu, logvar
        else:
            return h, None, None

    def forward(self, x, dz):
        h = x
        
        # Through encoder
        h, mu, logvar = self.encode(h)

        if self.intervene:
            # Through geom
            if not self.grp_morphism.learn_params:
                dz = dz * self.specified_step
            transformed_h = \
                self.grp_morphism.transform(h[:,:self.n_transform_units], dz)
            h = torch.hstack([transformed_h,h[:,self.n_transform_units:]])
        # Through decoder
        h = self.decoder(h)
        if self.variational:
            return h, mu, logvar
        else:
            return h, None, None