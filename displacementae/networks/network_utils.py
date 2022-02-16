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

from os import error
import torch

from networks.cnn import CNN
from networks.transposedcnn import TransposedCNN 
import networks.geometric.orthogonal as orth
import utils.misc as misc
import networks.autoencoder as ae
import networks.autoencoder_prodrep as aeprod

BLOCK_REPR = 'blockrepr'
PROD_REPR = 'prodrepr'
AUTOENCODER = 'autoencoder'

def setup_network(config, dhandler, device, mode=AUTOENCODER, 
                  repr=BLOCK_REPR):
    if mode==AUTOENCODER:
        return setup_autoencoder_network(config, dhandler, device, 
                                         repr)
    else:
        raise NotImplementedError

def setup_autoencoder_network(config, dhandler, device, repr):
    """
    Sets up an autoencoder with a geometric transformation of the latent units.
    
    The autoencoder consists of a contracting path, 
    a geometric transformation of the latent space and
    an expanding path back into the input space.

    Args:
        config (Namespace): configuration of the experiment, obtained from cli.
        dhandler (dataset): Handler for dataset.
        device (str): indicates device where parameters are stored.
        repr (str): Indicates which group representation to use for the 
                    observed actions. in ['block_repr','prod_repr']
                    if 'block_repr': group representation is block diagonal 
                        2D rotation matrices.
    """
    in_channels, shape_in = dhandler.in_shape[0], dhandler.in_shape[1:]
    conv_channels = [in_channels] + misc.str_to_ints(config.conv_channels)
    
    kernel_sizes = misc.str_to_ints(config.kernel_sizes)
    strides = misc.str_to_ints(config.strides)
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes[0]
    if len(strides) == 1:
        strides = strides[0]
    
    if isinstance(strides,list):
        trans_strides = strides[::-1]
    else:
        trans_strides = strides

    if isinstance(kernel_sizes, list):
        trans_kernel = kernel_sizes[::-1]
    else:
        trans_kernel = kernel_sizes
    
    n_free_units = config.n_free_units

    if repr == BLOCK_REPR:
        transformed_units = dhandler.action_shape[0] * 2
    elif repr == PROD_REPR:
        transformed_units = config.repr_dim

    repr_units =  transformed_units + n_free_units

    lin_channels = misc.str_to_ints(config.lin_channels)
    if config.net_act=='relu':
        act_fn = torch.relu
    elif config.net_act=='sigmoid':
        act_fn = torch.sigmoid
    elif config.net_act=='tanh':
        act_fn = torch.tanh
    else:
        act_fn = None

    variational = config.variational
    if variational and config.beta is None:
        config.beta = 1.

    if not hasattr(config,'specified_grp_step'):
        specified_step = 0
    else:
        specified_step = misc.str_to_floats(config.specified_grp_step)
        if len(specified_step) == 0 and not config.learn_geometry:
            raise ValueError
        if len(specified_step) == 1:
            specified_step = specified_step[0]


    # if variational, encoder outputs mean and logvar
    encoder_outputs = (1 + variational ) * repr_units 
    encoder = CNN(shape_in=shape_in, kernel_sizes=kernel_sizes, strides=strides,
        conv_channels=conv_channels,
        linear_channels=lin_channels+[encoder_outputs],
        use_bias=True, activation_fn=act_fn).to(device)
    
    decoder = TransposedCNN(shape_out=shape_in, kernel_sizes=trans_kernel,
        strides=trans_strides, conv_channels=conv_channels[::-1],
        linear_channels=[repr_units]+lin_channels[::-1],
        use_bias=True, activation_fn=act_fn).to(device)
    
    if repr == BLOCK_REPR:
        orthogonal_matrix = orth.OrthogonalMatrix(
            transformation=orth.OrthogonalMatrix.BLOCKS, 
            n_units=transformed_units, device=device, 
            learn_params=config.learn_geometry).to(device)
        
        autoencoder = ae.AutoEncoder(
            encoder=encoder,decoder=decoder, 
            grp_transformation=orthogonal_matrix,
            variational=variational,specified_step=specified_step, 
            n_repr_units=repr_units, intervene=config.intervene, 
            spherical=config.spherical)
    elif repr == PROD_REPR:
        autoencoder = aeprod.AutoencoderProdrep(encoder=encoder,decoder=decoder,
                n_actions=dhandler.n_actions,
                n_repr_units=repr_units, n_transform_units=transformed_units,
                variational=variational,device = device, 
                spherical=config.spherical).to(device)

        
    else:
        raise NotImplementedError

    return autoencoder