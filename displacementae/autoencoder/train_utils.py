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
# @title          :displacementae/autoencoder/train_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/11/2021
# @version        :1.0
# @python_version :3.7.4

from argparse import Namespace
import torch
import torch.nn as nn

import autoencoder.train_args as train_args
import data.data_utils as data_utils
import networks.network_utils as net_utils
import utils.sim_utils as sim_utils
import networks.variational_utils as var_utils

def setup_optimizer(params, config):
    lr = config.lr
    weight_decay = config.weight_decay
    if config.use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def train(dhandler, dloader, nets, config, shared, device, logger, mode):
    n_actions = dhandler.action_shape[0]

    params = []
    for net in nets:
        params += list(net.parameters())
    if mode == 'autoencoder':
        encoder,decoder,orthog_mat = nets

    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False

    for epoch in range(epochs):
        logger.info(f"Training epoch {epoch}.")
        for i, batch in enumerate(dloader):
            optim.zero_grad()
            x1, y1, x2, y2, dj = [a.to(device) for a in batch]
            x1 = x1.unsqueeze(1).float()
            x2 = x2.unsqueeze(1).float()
            dj = dj.float()
            ### Forward ###
            # Through encoder
            h = encoder(x1)
            mu, logvar = h[:,2*n_actions:],h[:,2*n_actions:]
            h = var_utils.reparametrize(mu,logvar)
            # Through geom
            h = orthog_mat.rotate(h,dj[:,dhandler.intervened_on])
            # Through decoder
            x2_hat = torch.sigmoid(decoder(h))

            logger.info(f'learned alpha {orthog_mat.alpha}')
            ### Losses
            # KL
            kl_loss = var_utils.kl_loss(mu,logvar)
            # Reconstruction
            bce_loss = nn.BCELoss(reduction='sum')(x2_hat,x2)
            total_loss = kl_loss + bce_loss
            logger.info(f'Total loss {total_loss} \t=BCE \t{bce_loss} \t+KL \t{kl_loss}')
            total_loss.backward()
            optim.step()
    return interrupted_training


def run(mode='autoencoder'):
    # parse commands
    config = train_args.parse_cmd_arguments()
    # setup environment
    device, logger = sim_utils.setup_environment(config)
    sim_utils.backup_cli_command(config)
    # setup dataset
    dhandler, dloader = data_utils.setup_data(config)
    # setup models
    nets = net_utils.setup_network(config, dhandler, device, mode=mode)
    # setup shared
    shared = Namespace()

    logger.info('### Training ###')
    finished_training = train(dhandler, dloader, nets,
                              config, shared, device, logger, mode)
    return
