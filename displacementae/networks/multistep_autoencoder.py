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
# @title          :displacementae/networks/multistep_autoencoder.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :21/03/2022
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn.functional as F

from networks.autoencoder import AutoEncoder

class MultistepAutoencoder(AutoEncoder):
    def __init__(self, encoder, decoder, grp_morphism, n_repr_units, 
                 n_transform_units, variational=True, spherical=False,
                 normalize_post_act=False):        
        """
        An Autoencoder with multiple future observation prediction through 
        group representation.

        encoder, nn.Module: Encoder Network. Maps inputs to a representation 
                        vector or to parameters of a posterior distribution 
                        in the variational case. 
        decoder, nn.Module: Decoder Network. Maps a representation 
                        vector back in the input space. 
        grp_transformation, nn.Module: Maps an action to a transformation of
                        the representation space.
        variational, bool: If True, the encoder describes a distribution instead 
                        of being deterministic, defaults to True.
        spherical, bool: If True, the encoder's outputs (the location part 
                        in the variational case) is normalized.
        """
        super().__init__(encoder=encoder,decoder=decoder,
                         grp_morphism=grp_morphism, n_repr_units=n_repr_units,
                         variational=variational, spherical=spherical)
        self.n_transform_units = n_transform_units
        self.normalize_post_act = normalize_post_act

    def forward(self, x, dz):
        """
        Encodes the input image and predicts the `n_steps` following images.

        Encodes the input image `x`. Decodes each image after the ith observed 
        transition. Transitions `dz` are mapped to matrices through the 
        `grp_morphism`, matrices are applied to the obtained representation.
        """
        h = x
        n_steps = dz.shape[1]
    
        # Through encoder
        h, mu, logvar = self.encode(h)

        h_out = torch.empty(
            size=[x.shape[0]] + [n_steps, self.n_repr_units],device=x.device)

        if self.n_repr_units > self.n_transform_units:
            # The part of the transformation that is not transformed 
            # is repeated for all transition steps.
            h_out[:,:,self.n_transform_units:] = \
                                    h[:,self.n_transform_units:]\
                                        .unsqueeze(1).repeat(1,n_steps,1)
        # Through geometry
        for i in range(n_steps):
            if i == 0:
                h_out[:,i,:self.n_transform_units] = \
                    self.grp_morphism.act(dz[:,i], 
                                          h[:,:self.n_transform_units])   
            else:
                h_out[:,i,:self.n_transform_units] = \
                        self.grp_morphism.act(
                                dz[:,i], 
                                h_out[:,i-1,:self.n_transform_units].clone())
                                
        # Through decoder
        h_out = h_out.view(-1, self.n_repr_units)
        h_out = self.decoder(h_out)
        h_out = h_out.view(x.shape[0],n_steps,*x.shape[1:])
        if self.variational:
            return h_out, mu, logvar
        else:
            return h_out, None, None
