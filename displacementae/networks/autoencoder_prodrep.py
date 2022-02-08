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
# @title          :displacementae/networks/autoencoder_prodrep.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :07/02/2022
# @version        :1.0
# @python_version :3.7.4

import torch

import networks.autoencoder as ae
import networks.geometric.prodrepr.action_lookup as al
import networks.variational_utils as var_utils

class AutoencoderProdrep(ae.AutoEncoder):
    def __init__(self,encoder, decoder, n_actions, n_repr_units, 
                 n_transform_units, variational=True):
        super().__init__(encoder, decoder, variational=variational, 
                n_repr_units = n_repr_units, grp_transformation = None, 
                specified_step=0, intervene=True)
        self.n_transform_units = n_transform_units
        self.n_actions = n_actions
        self.grp_transform = al.ActionLookup(self.n_actions,
                                             dim=self.n_transform_units)

    def forward(self, x, a):
        h = x
        # Through encoder
        h = self.encoder(x)
        if self.variational:
            half = h.shape[1]//2
            mu, logvar = h[:, : half], h[:, half:]
            h = var_utils.reparametrize(mu, logvar)
        # Through geometry
        temp = []
        for i in range(x.shape[0]):
            temp.append(
                self.grp_transform.forward(h[i,:self.n_transform_units], 
                                           a[i].squeeze()))    
        
        transformed_h = torch.vstack(temp)
        if self.n_transform_units < self.n_repr_units:
            h = torch.hstack([transformed_h,h[:,self.n_transform_units:]])
        else:
            h = transformed_h
        # Through decoder
        h = self.decoder(h)
        if self.variational:
            return h, mu, logvar
        else:
            return h, None, None