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

import torch

from networks.cnn import CNN
from networks.transposedcnn import TransposedCNN 
import networks.geometric.orthogonal as orth
import utils.misc as misc
from networks.autoencoder import AutoEncoder

def setup_network(config, dhandler, device, mode='autoencoder'):
    if mode=='autoencoder':
        return setup_autoencoder_network(config, dhandler, device)
    elif mode=='contrastive':
        return

def setup_autoencoder_network(config, dhandler, device):
    """
    Sets up an autoencoder with a geometric transformation of the latent units.
    
    The autoencoder consists of a contracting path, 
    a geometric transformation of the latent space and
    an expanding path back into the input space. 
    """
    in_channels, shape_in = dhandler.in_shape[0], dhandler.in_shape[1:]
    kernel_sizes = misc.str_to_ints(config.kernel_sizes)
    strides = misc.str_to_ints(config.strides)
    if len(kernel_sizes) == 1:
        kernel_sizes = kernel_sizes[0]
    if len(strides) == 1:
        strides = strides[0]
             
    conv_channels = [in_channels] + misc.str_to_ints(config.conv_channels)
    latent_units = dhandler.action_shape[0] * 2
    lin_channels = misc.str_to_ints(config.lin_channels)
    if config.net_act=='relu':
        act_fn = torch.relu
    elif config.net_act=='sigmoid':
        act_fn = torch.sigmoid
    elif config.net_act=='tanh':
        act_fn = torch.tanh
    else:
        act_fn = None

    if not hasattr(config,'variational'):
        variational = 1
        config.variational = True
    else:
        variational = config.variational
    
    if not hasattr(config,'specified_grp_step'):
        specified_step = 0
    else:
        specified_step = config.specified_step


    # if variational, encoder outputs mean and logvar
    encoder_outputs = (1 + variational ) * latent_units 
    encoder = CNN(shape_in=shape_in, kernel_sizes=kernel_sizes, strides=strides,
        conv_channels=conv_channels,
        linear_channels=lin_channels+[encoder_outputs],
        use_bias=True, activation_fn=act_fn).to(device)
    
    decoder = TransposedCNN(shape_out=shape_in, kernel_sizes=kernel_sizes,
        strides=strides, conv_channels=conv_channels[::-1],
        linear_channels=[latent_units]+lin_channels[::-1],
        use_bias=True, activation_fn=act_fn).to(device)
    
    orthogonal_matrix = orth.OrthogonalMatrix(
        transformation=orth.OrthogonalMatrix.BLOCKS, n_units=latent_units, 
        device=device, learn_params=config.learn_geometry).to(device)
    
    autoencoder = AutoEncoder(
        encoder=encoder,decoder=decoder, grp_transformation=orthogonal_matrix,
        variational=variational,specified_step=specified_step)

    return autoencoder