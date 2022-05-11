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
# @title          :displacementae/networks/autoencoder_prodrep.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :07/02/2022
# @version        :1.0
# @python_version :3.7.4

import torch

import networks.autoencoder as ae
import grouprepr.prodrepr.action_lookup as al
import networks.variational_utils as var_utils
import utils.data_utils as udutils


class AutoencoderProdrep(ae.AutoEncoder):
    """An autoencoder with a transformation of the latent from a known action.
    The transformation is a matrix multiplication with an SO(n) matrix
    represented by a product of matrices of 2D rotations.

    Args:
        encoder (`torch.nn.Module`): Maps the input to a latent representation,
                or, if :param:variational is `True` to a posterior distribution.
        decoder (`torch.nn.Module`): Maps from the latent space back to the
                input space.
        n_actions (int): Number of possible actions.
        n_repr_units (int): Total number of representation units.
        n_transform_units (int): Number of representation units acted on by the 
                action representation.
        variational (bool): Whether this is a variational autoencoder. 
                If `True`, the :method:`forward` outputs the reconstruction, 
                the mean and logvar.  
    """
    def __init__(self,encoder, decoder, n_actions, n_repr_units, 
                 n_transform_units, variational=True, device='cpu', 
                 spherical=False):
        """Constructor Method
        """
        super().__init__(encoder, decoder, variational=variational, 
                n_repr_units = n_repr_units, grp_transformation = None, 
                specified_step=0, intervene=True, spherical=spherical)
        self.n_transform_units = n_transform_units
        self.n_actions = n_actions
        self.grp_morphism = al.ActionLookup(self.n_actions,
                        dim_representation=self.n_transform_units,device=device)

    def forward(self, x, a):
        """
        encodes input signals, transforms with input actions then decodes.

        Args:
            x (ndarray): input signal.
            a (ndarray): input actions. Each action is an int in the range 
                        (0,n_actions). TODO input is a normal acion converted within function call?


        Returns:
            (ndarray): If :param:`variational` is `False`: 
                            output of the decoder. 
                       If :param:`variational` is `True`:
                            output of decoder, mu, logvar.
        """
        h = x
        # Through encoder
        h, mu, logvar = self.encode(h)

        # Through geometry
        a = udutils.action_to_id(a)
        temp = []
        for i in range(x.shape[0]):
            temp.append(
                self.grp_morphism.forward(h[i,:self.n_transform_units], 
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

    def entanglement_loss(self):
        return self.grp_morphism.entanglement_loss()
