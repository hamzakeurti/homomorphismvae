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
# @title          :displacementae/networks/network_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :12/11/2021
# @version        :1.0
# @python_version :3.7.4

from argparse import Namespace
from os import error
import torch
import torch.nn as nn
from displacementae.data.transition_dataset import TransitionDataset

from displacementae.networks.cnn import CNN
from displacementae.networks.transposedcnn import TransposedCNN 
import displacementae.grouprepr.blockrots.orthogonal as orth
import displacementae.utils.misc as misc
from displacementae.networks.autoencoder import AutoEncoder
from displacementae.networks.multistep_autoencoder import MultistepAutoencoder
from displacementae.grouprepr.block_mlp_representation import BlockMLPRepresentation
from displacementae.grouprepr.group_representation import GroupRepresentation
from displacementae.grouprepr.prodrepr.action_lookup import ActionLookup
from displacementae.grouprepr.representation_utils import Representation, str_to_enum
from displacementae.grouprepr.lookup_representation import LookupRepresentation
from displacementae.grouprepr.block_lookup_representation import BlockLookupRepresentation
from displacementae.grouprepr.trivial_representation import TrivialRepresentation
from displacementae.grouprepr.mlp_representation import MLPRepresentation
from displacementae.grouprepr.unstructured_representation import UnstructuredRepresentation
from displacementae.grouprepr.soft_block_mlp_representation import SoftBlockMLPRepresentation
from displacementae.grouprepr.varphi import VarPhi

AUTOENCODER = 'autoencoder'


def setup_network(config, dhandler, device, mode=AUTOENCODER):
    if mode == AUTOENCODER:
        return setup_autoencoder_network(config, dhandler, device)
    elif mode == 'homomorphism':
        return setup_multistep_autoencoder(config, dhandler, device)
    elif mode == 'trajectory':
        return setup_multistep_autoencoder(config, dhandler[0], device)
    elif mode == 'supervised':
        repr_units = dhandler.action_units
        if config.net_mode == 'encoder':
            return setup_encoder(config,dhandler,device,repr_units)
        elif config.net_mode == 'decoder':
            return setup_decoder(config,dhandler,device,repr_units)
        elif config.net_mode == 'grouprepr':
            grp_morphism = setup_grp_morphism(config, dhandler, device)
            d = grp_morphism.dim_representation
            a = dhandler.action_units
            if d**2 != a:
                raise ValueError(f"Group morphism outputs {d**2} units = " +
                            f"({d}x{d}), expected {a}.")
            nets = nn.Sequential(grp_morphism, nn.Flatten())
            return nets
    else:
        raise NotImplementedError



def setup_encoder(config,dhandler,device,repr_units):
    in_channels, shape_in = dhandler.in_shape[0], dhandler.in_shape[1:]
    conv_channels = [in_channels] + misc.str_to_ints(config.conv_channels)

    kernel_sizes = misc.str_to_ints(config.kernel_sizes)
    strides = misc.str_to_ints(config.strides)
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes[0]
    if len(strides) == 1:
        strides = strides[0]


    lin_channels = misc.str_to_ints(config.lin_channels)
    if config.net_act == 'relu':
        act_fn = torch.relu
    elif config.net_act == 'sigmoid':
        act_fn = torch.sigmoid
    elif config.net_act == 'tanh':
        act_fn = torch.tanh
    else:
        act_fn = None

    variational = config.variational
    if variational and config.beta is None:
        config.beta = 1.

    # if variational, encoder outputs mean and logvar
    encoder_outputs = (1 + variational) * repr_units
    encoder = CNN(shape_in=shape_in, kernel_sizes=kernel_sizes,
        strides=strides, conv_channels=conv_channels,
        linear_channels=lin_channels+[encoder_outputs],
        use_bias=True, activation_fn=act_fn).to(device)

    return encoder

def setup_decoder(config,dhandler,device,repr_units):
    in_channels, shape_in = dhandler.in_shape[0], dhandler.in_shape[1:]
    
    lin_channels = [repr_units]
    if config.decoder_conv_channels == "-1":
        conv_channels = misc.str_to_ints(config.conv_channels)[::-1]
        lin_channels += misc.str_to_ints(config.lin_channels)[::-1]
        kernel_sizes = misc.str_to_ints(config.kernel_sizes)[::-1]
        strides = misc.str_to_ints(config.strides)[::-1]
    else:
        conv_channels = misc.str_to_ints(config.decoder_conv_channels)
        lin_channels += misc.str_to_ints(config.decoder_lin_channels)
        kernel_sizes = misc.str_to_ints(config.decoder_kernel_sizes)
        strides = misc.str_to_ints(config.decoder_strides)
    conv_channels += [in_channels]
    
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes[0]
    if len(strides) == 1:
        strides = strides[0]


    if config.net_act == 'relu':
        act_fn = torch.relu
    elif config.net_act == 'sigmoid':
        act_fn = torch.sigmoid
    elif config.net_act == 'tanh':
        act_fn = torch.tanh
    else:
        act_fn = None

    variational = config.variational
    if variational and config.beta is None:
        config.beta = 1.
    
    decoder = TransposedCNN(shape_out=shape_in, kernel_sizes=kernel_sizes,
                        strides=strides, conv_channels=conv_channels,
                        linear_channels=lin_channels,
                        use_bias=True, activation_fn=act_fn).to(device)
    return decoder

def setup_encoder_decoder(config, dhandler, device, repr_units):
    # in_channels, shape_in = dhandler.in_shape[0], dhandler.in_shape[1:]
    # conv_channels = [in_channels] + misc.str_to_ints(config.conv_channels)

    # kernel_sizes = misc.str_to_ints(config.kernel_sizes)
    # strides = misc.str_to_ints(config.strides)
    # if len(kernel_sizes) == 1:
    #     kernel_sizes = kernel_sizes[0]
    # if len(strides) == 1:
    #     strides = strides[0]

    # if isinstance(strides, list):
    #     trans_strides = strides[::-1]
    # else:
    #     trans_strides = strides

    # if isinstance(kernel_sizes, list):
    #     trans_kernel = kernel_sizes[::-1]
    # else:
    #     trans_kernel = kernel_sizes

    # lin_channels = misc.str_to_ints(config.lin_channels)
    # if config.net_act == 'relu':
    #     act_fn = torch.relu
    # elif config.net_act == 'sigmoid':
    #     act_fn = torch.sigmoid
    # elif config.net_act == 'tanh':
    #     act_fn = torch.tanh
    # else:
    #     act_fn = None

    # variational = config.variational
    # if variational and config.beta is None:
    #     config.beta = 1.

    # # if variational, encoder outputs mean and logvar
    # encoder_outputs = (1 + variational) * repr_units
    # encoder = CNN(shape_in=shape_in, kernel_sizes=kernel_sizes,
    #                 strides=strides, conv_channels=conv_channels,
    #                 linear_channels=lin_channels+[encoder_outputs],
    #                 use_bias=True, activation_fn=act_fn).to(device)

    # decoder = TransposedCNN(shape_out=shape_in, kernel_sizes=trans_kernel,
    #     strides=trans_strides, conv_channels=conv_channels[::-1],
    #     linear_channels=[repr_units]+lin_channels[::-1],
    #     use_bias=True, activation_fn=act_fn).to(device)
    encoder = setup_encoder(config, dhandler, device, repr_units)
    decoder = setup_decoder(config, dhandler, device, repr_units)

    return encoder, decoder


def setup_grp_morphism(config: Namespace, dhandler: TransitionDataset,
                       device: str
                       ) -> GroupRepresentation:
    """
    Sets up the group morphism module which converts input actions to
    """

    varphi_units = misc.str_to_ints(config.varphi_units)
    varphi = VarPhi(n_action_units=dhandler.action_units,
           device=device, 
           linear_units=varphi_units,
           activation=config.varphi_act,
           seed=config.varphi_random_seed,
           ).to(device)
    representation = str_to_enum(config.grouprepr)

   
    if representation == Representation.MLP:
        hidden_units = misc.str_to_ints(config.group_hidden_units)
        grp_morphism = MLPRepresentation(
                n_action_units=dhandler.action_units,
                dim_representation=config.dim,
                hidden_units=hidden_units,device=device, 
                normalize=config.normalize,
                normalize_post_action=config.normalize_post_action,
                exponential_map=config.exponential_map,
                varphi=varphi
                ).to(device)

    elif representation == Representation.BLOCK_MLP:
        dims = misc.str_to_ints(config.dims)
        hidden_units = misc.str_to_ints(config.group_hidden_units)
        grp_morphism = BlockMLPRepresentation(
                n_action_units=dhandler.action_units,
                dim_representation=sum(dims),
                dims=dims,
                hidden_units=hidden_units,
                device=device,
                normalize_subrepresentations=config.normalize_subrepresentations,
                normalize_post_action=config.normalize_post_action,
                exponential_map=config.exponential_map,
                varphi=varphi
                ).to(device)

    elif representation == Representation.SOFT_BLOCK_MLP:
        hidden_units = misc.str_to_ints(config.group_hidden_units)
        grp_morphism = SoftBlockMLPRepresentation(
                n_action_units=dhandler.action_units,
                dim_representation=config.dim,
                hidden_units=hidden_units,device=device, 
                normalize=config.normalize,
                normalize_post_action=config.normalize_post_action,
                exponential_map=config.exponential_map,
                varphi=varphi,
                regularize_algebra=config.regularize_algebra).to(device)

    elif representation == Representation.PROD_ROTS_LOOKUP:
        grp_morphism = ActionLookup(
                n_action_units=dhandler.n_actions,
                dim_representation=config.dim,
                repr_loss_on=True,
                repr_loss_weight=config.grp_loss_weight,
                device=device,
                varphi=varphi,
                ).to(device)

    elif representation == Representation.BLOCK_ROTS:
        if not hasattr(config, 'specified_grp_step'):
            specified_step = 0
        else:
            specified_step = misc.str_to_floats(config.specified_grp_step)
            if len(specified_step) == 0 and not config.learn_geometry:
                raise ValueError
            if len(specified_step) == 1:
                specified_step = specified_step[0]

        grp_morphism = orth.OrthogonalMatrix(
            dim_representation=dhandler.action_units * 2, device=device,
            learn_params=config.learn_geometry,
            specified_step=specified_step,
            varphi=varphi,).to(device)

    elif representation == Representation.LOOKUP:
        grp_morphism = LookupRepresentation(
                n_actions=dhandler.n_actions,
                dim_representation=config.dim,
                device=device,
                normalize=config.normalize,
                normalize_post_action=config.normalize_post_action,
                exponential_map=config.exponential_map,
                varphi=varphi,
                ).to(device)

    elif representation == Representation.BLOCK_LOOKUP:
        dims = misc.str_to_ints(config.dims)
        grp_morphism = BlockLookupRepresentation(
                n_actions=dhandler.n_actions,
                dims=dims,
                dim_representation=sum(dims),
                device=device,
                normalize_subrepresentations=config.normalize_subrepresentations,
                normalize_post_action=config.normalize_post_action,
                exponential_map=config.exponential_map,
                varphi=varphi).to(device)

    elif representation == Representation.TRIVIAL:
        grp_morphism = TrivialRepresentation(
                dim_representation=config.dim,
                device=device,
                varphi=varphi).to(device)

    elif representation == Representation.UNSTRUCTURED:
        grp_morphism = UnstructuredRepresentation(
                n_action_units=dhandler.n_actions,
                dim_representation=config.dim,
                hidden_units=config.group_hidden_units,
                device=device,
                varphi=varphi).to(device)

            
    else:
        raise NotImplementedError(
                f'Representation {representation} is not implemented.')
    return grp_morphism




def setup_autoencoder_network(config, dhandler, device):
    """
    Sets up an autoencoder with a geometric transformation of the latent
    units.

    The autoencoder consists of a contracting path,
    a geometric transformation of the latent space and
    an expanding path back into the input space.

    Args:
        config (Namespace): configuration of the experiment, obtained
                            from cli.
        dhandler (dataset): Handler for dataset.
        device (str): indicates device where parameters are stored.
        representation (str): Indicates which group representation to use for the
                    observed actions.
                    if 'block_repr': group representation is block
                    diagonal 2D rotation matrices.
    """
    grp_morphism = setup_grp_morphism(config, device=device, dhandler=dhandler)

    dim_representation = grp_morphism.dim_representation
    n_free_units = config.n_free_units
    repr_units = dim_representation + n_free_units

    encoder, decoder = setup_encoder_decoder(config, dhandler, device,
                                             repr_units)

    autoencoder = AutoEncoder(
        encoder=encoder,decoder=decoder, grp_morphism=grp_morphism,
        variational=config.variational, n_repr_units=repr_units, 
        intervene=config.intervene, spherical=config.spherical)
    return autoencoder


def setup_multistep_autoencoder(config, dhandler, device):
    grp_morphism = setup_grp_morphism(config, device=device, dhandler=dhandler)

    dim_representation = grp_morphism.dim_representation
    n_free_units = config.n_free_units
    repr_units = dim_representation + n_free_units

    encoder, decoder = setup_encoder_decoder(config, dhandler, device,
                                             repr_units)

    autoencoder = MultistepAutoencoder(
                        encoder=encoder, decoder=decoder,
                        grp_morphism=grp_morphism,
                        variational=config.variational,
                        n_repr_units=repr_units,
                        n_transform_units=dim_representation,
                        spherical=config.spherical,
                        spherical_post_action=config.spherical_post_action,
                        reconstruct_first=config.reconstruct_first)

    return autoencoder
