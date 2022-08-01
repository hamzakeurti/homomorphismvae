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
                 reconstruct_first=False, spherical_post_action=False):        
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
        self.reconstruct_first = reconstruct_first
        self.spherical_post_action = spherical_post_action

    def forward(self, x, dz):
        """
        Encodes the input image and predicts the `n_steps` following images.

        Encodes the input image `x`. Decodes each image after the ith observed
        transition. Transitions `dz` are mapped to matrices through the
        `grp_morphism`, matrices are applied to the obtained representation.
        """
        h = x
        if dz is None:
            n_steps = 0
        else:
            n_steps = dz.shape[1]

        n_images = n_steps
        if self.reconstruct_first:
            n_images += 1
    
        # Through encoder
        h, mu, logvar = self.encode(h)
        latent = h

        h_out = self.act(h,dz)

        h_out = torch.empty(
            size=[x.shape[0]] + [n_images, self.n_repr_units],device=x.device)

        if self.n_repr_units > self.n_transform_units:
            # The part of the transformation that is not transformed
            # is repeated for all transition steps.
            h_out[:,:,self.n_transform_units:] = \
                                    h[:,self.n_transform_units:]\
                                        .unsqueeze(1).repeat(1,n_images,1)


        # Normalize the encoder's output according to subspaces of 
        # the group representation.
        h[:,:self.n_transform_units] = \
                self.grp_morphism.normalize_vector(
                    h[:,:self.n_transform_units].clone())

        if self.reconstruct_first:
            h_out[:,0,...] = h.clone()
        else:
            h_out[:,0,:self.n_transform_units] = self.grp_morphism.act(
                                      dz[:,0], 
                                      h[:,:self.n_transform_units])   

        # Through geometry
        for i in range(1,n_steps):
            h_out[:,i,:self.n_transform_units] = \
                    self.grp_morphism.act(
                            dz[:,i], 
                            h_out[:,i-1,:self.n_transform_units].clone())

        if self.spherical_post_action:
            h_out[...,:self.n_transform_units] =\
                 F.normalize(h_out[...,:self.n_transform_units].clone(),dim=-1)            

        # Through decoder
        latent_hat = h_out
        h_out = h_out.view(-1, self.n_repr_units)
        h_out = self.decoder(h_out)
        h_out = h_out.view(x.shape[0],n_images,*x.shape[1:])
        if self.variational:
            return h_out, latent, latent_hat, mu, logvar
        else:
            return h_out, latent, latent_hat, None, None


    def forward2(self, x, dz):
        """
        Encodes the input image and predicts the `n_steps` following images.

        Encodes the input image `x`. Decodes each image after the ith observed
        transition. Transitions `dz` are mapped to matrices through the
        `grp_morphism`, matrices are applied to the obtained representation.
        """
        h = x
        if dz is None:
            n_steps = 0
        else:
            n_steps = dz.shape[1]

        n_images = n_steps
        if self.reconstruct_first:
            n_images += 1

        # Through encoder
        h, mu, logvar = self.encode(h)

        h_out = self.act(h, dz)

        h_out = self.decode(h_out)

        if self.variational:
            return h_out, mu, logvar
        else:
            return h_out, None, None

    def act(self, h, dz):
        """
        Forwards latent vectors through the group representation of input
        transitions.

        """
        n_steps = dz.shape[1]

        h_out = torch.empty(
            size=[dz.shape[0]] + [n_steps + 1, self.n_repr_units], device=dz.device)

        if self.n_repr_units > self.n_transform_units:
            # The part of the transformation that is not transformed
            # is repeated for all transition steps.
            h_out[:, :, self.n_transform_units:] = \
                h[:, None, self.n_transform_units:].repeat(1, n_steps + 1, 1)

        # Normalize the encoder's output according to subspaces of
        # the group representation.
        h[:, :self.n_transform_units] = \
            self.grp_morphism.normalize_vector(
                    h[:, :self.n_transform_units].clone())

        h_out[:, 0] = h.clone()

        # Through geometry
        for i in range(n_steps):
            h_out[:, i + 1, :self.n_transform_units] = \
                    self.grp_morphism.act(
                            dz[:, i],
                            h_out[:, i, :self.n_transform_units].clone())

        if self.spherical_post_action:
            h_out[..., :self.n_transform_units] =\
                 F.normalize(h_out[..., :self.n_transform_units].clone(), dim=-1)

        if self.reconstruct_first:
            return h_out
        else:
            return h_out[:, 1:]

    def decode(self, h):
        """
        Forwards a sequence of latent vectors through the decoder.
        Outputs images
        """
        # Through decoder
        h_out = h.reshape(-1, self.n_repr_units)
        h_out = self.decoder(h_out)
        h_out = h_out.reshape(h.shape[0],h.shape[1],*h_out.shape[1:])
        return h_out


    def normalize_representation(self,z:torch.Tensor) -> torch.Tensor:
        """
        Normalize subrepresentation spaces according to the group action.

        If the group representation is a direct sum of subrepresentations,
        then each subrepresentation is normalized individually.
        """
        z_out = z
        z_out[:,:self.n_transform_units] = \
              self.grp_morphism.normalize_vector(z[:,:self.n_transform_units])
        return z_out
