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
import os
import torch
import torch.nn as nn

import autoencoder.train_args as train_args
import data.data_utils as data_utils
import networks.network_utils as net_utils
import networks.variational_utils as var_utils
import utils.plotting_utils as plt_utils
import utils.sim_utils as sim_utils
import utils.misc as misc


def setup_optimizer(params, config):
    lr = config.lr
    weight_decay = config.weight_decay
    if config.use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def evaluate(dhandler, nets, device, config, shared, logger, mode, epoch,
             save_fig=False, plot=False):
    nets.eval()
    with torch.no_grad():
        img1, cls1, img2, cls2, dj = dhandler.get_val_batch()
        X1 = torch.FloatTensor(img1).to(device)
        X2 = torch.FloatTensor(img2).to(device)
        dj = torch.FloatTensor(dj).to(device)
        h, mu, logvar = nets(X2, dj[:, dhandler.intervened_on])
        X2_hat = torch.sigmoid(h)
        kl_loss = var_utils.kl_loss(mu, logvar)
        # Reconstruction
        bce_loss = nn.BCELoss(reduction='sum')(X2_hat, X2)
        total_loss = kl_loss + bce_loss
        logger.info(f'EVALUATION prior to epoch [{epoch}]...') 
        logger.info(f'[{epoch}] loss\t{total_loss.item():.2f} =\t' +
            f'BCE {bce_loss.item():.2f} +\tKL {kl_loss.item():.5f}')
        if plot:
            fig_dir = os.path.join(config.out_dir, 'figures')
            figname = None
            if save_fig:
                figname = os.path.join(fig_dir, f'{epoch}_')
            plt_utils.plot_reconstruction(dhandler, nets, shared, config, device,
                                        logger, mode, figname)
            vary_joints = misc.str_to_ints(config.plot_vary_joints)
            plot_latent = misc.str_to_ints(config.plot_manifold_latent)
            if len(plot_latent) > 0:
                plt_utils.plot_manifold(dhandler, nets, shared, config, device,
                                        logger, mode, epoch, vary_joints=vary_joints,
                                        plot_latent=plot_latent, figname=figname)


def train(dhandler, dloader, nets, config, shared, device, logger, mode):
    n_actions = dhandler.action_shape[0]

    params = nets.parameters()

    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False

    for epoch in range(epochs):
        if epoch % config.val_epoch == 0:
            evaluate(dhandler, nets, device, config, shared, logger, mode,
                     epoch, save_fig=True, plot=not config.no_plots)
        logger.info(f"Training epoch {epoch}.")

        for i, batch in enumerate(dloader):
            optim.zero_grad()
            x1, y1, x2, y2, dj = [a.to(device) for a in batch]
            x1 = x1.unsqueeze(1).float()
            x2 = x2.unsqueeze(1).float()
            dj = dj.float()
            ### Forward ###
            # Through encoder
            h, mu, logvar = nets(x1, dj[:, dhandler.intervened_on])

            x2_hat = torch.sigmoid(h)

            logger.info(f'learned alpha {nets.grp_transform.alpha.item()}')
            # Losses
            # KL
            kl_loss = var_utils.kl_loss(mu, logvar)
            # Reconstruction
            bce_loss = nn.BCELoss(reduction='sum')(x2_hat, x2)
            total_loss = kl_loss + bce_loss
            total_loss.backward()
            optim.step()
            logger.info(f'[{epoch}:{i}] loss\t{total_loss.item():.2f} =\t' +
                        f'BCE {bce_loss.item():.2f} +\tKL {kl_loss.item():.5f}')
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
