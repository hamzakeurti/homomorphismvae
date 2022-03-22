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
# @title          :displacementae/homomorphism/train_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/03/2022
# @version        :1.0
# @python_version :3.7.4

from argparse import Namespace
import os
import torch
import torch.nn as nn

import data.data_utils as data_utils
import networks.network_utils as net_utils
import networks.variational_utils as var_utils
import utils.plotting_utils as plt_utils
import utils.sim_utils as sim_utils
import utils.misc as misc
import utils.checkpoint as ckpt


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
    if epoch == 0:
        shared.bce_loss1 = []
        shared.bce_loss2 = []
        if config.variational:
            shared.kl_loss = []
    if epoch % config.val_epoch == 0:

        with torch.no_grad():
            img1, cls1, img2, cls2, dj1, img3, cls3, dj2 = dhandler.get_val_batch()
            X1 = torch.FloatTensor(img1).to(device)
            X2 = torch.FloatTensor(img2).to(device)
            X3 = torch.FloatTensor(img3).to(device)
            dj1 = torch.FloatTensor(dj1).to(device)
            dj2 = torch.FloatTensor(dj2).to(device)
            
            ### Forward ###
            # First reconstruction
            h1, mu1, logvar1 = nets(X1, dj1[:, dhandler.intervened_on])
            x2_hat = torch.sigmoid(h1)
            # Second reconstruction
            h2, mu2, logvar2 = nets(X1, 
                [dj1[:, dhandler.intervened_on],dj2[:, dhandler.intervened_on]])
            x3_hat = torch.sigmoid(h2)
            ### Losses
            # Reconstruction
            bce_loss1 = var_utils.bce_loss(x2_hat, X2)
            bce_loss2 = var_utils.bce_loss(x3_hat, X3)
            if nets.variational:
                # KL
                kl_loss = var_utils.kl_loss(mu1, logvar1)
                total_loss = config.beta * kl_loss + bce_loss1 + bce_loss2
            else:
                total_loss = bce_loss1 + bce_loss2

            ### Logging
            logger.info(f'EVALUATION prior to epoch [{epoch}]...') 
            log_text = f'[{epoch}] loss\t{total_loss.item():.2f}'
            log_text += f'=\tBCE1 {bce_loss1.item():.2f} '
            log_text += f'=\tBCE2 {bce_loss2.item():.2f} '

            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f}'
            logger.info(log_text)
            # Losses
            shared.bce_loss1.append(bce_loss1.item())
            shared.bce_loss2.append(bce_loss2.item())
            if nets.variational:
                shared.kl_loss.append(kl_loss.item())
            if nets.grp_transform.learn_params:
                alpha = nets.grp_transform.alpha.cpu().data.numpy().astype(float)
                logger.info(f'learned alpha {alpha}')
                if not hasattr(shared,"learned_alpha"):
                    shared.learned_alpha = []
                shared.learned_alpha.append(list(alpha))
            if epoch % 20*config.val_epoch == 0:
                sim_utils.save_dictionary(shared,config)

    if plot and (epoch % config.plot_epoch == 0):
        with torch.no_grad():
            fig_dir = os.path.join(config.out_dir, 'figures')
            figname = None
            if save_fig:
                figname = os.path.join(fig_dir, f'{epoch}_')
                shared.figname=figname
            plt_utils.plot_reconstruction(dhandler, nets, config, device,
                                        logger, figname)
            vary_latents = misc.str_to_ints(config.plot_vary_latents)
            plot_latent = misc.str_to_ints(config.plot_manifold_latent)
            if len(plot_latent) > 0:
                if not isinstance(plot_latent[0],list):
                    plot_latent = [plot_latent]
                    vary_latents = [vary_latents]
                for i in range(len(vary_latents)):
                    if config.plot_pca:
                        plt_utils.plot_manifold_pca(dhandler, nets, shared, config, 
                                            device, logger, mode, epoch, 
                                            vary_latents=vary_latents[i],
                                            figname=figname)    
                    else:
                        plt_utils.plot_manifold(dhandler, nets, shared, config, 
                                        device, logger, mode, epoch, 
                                        vary_latents=vary_latents[i],
                                        plot_latent=plot_latent[i], 
                                        figname=figname)
    nets.train()

def train(dhandler, dloader, nets, config, shared, device, logger, mode):
    params = nets.parameters()
    
    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False
    for epoch in range(epochs):
        evaluate(dhandler, nets, device, config, shared, logger, mode,
                    epoch, save_fig=True, plot=not config.no_plots)
        logger.info(f"Training epoch {epoch}.")

        for i, batch in enumerate(dloader):
            optim.zero_grad()
            x1, y1, x2, y2, dj1, x3, y3, dj2 = [a.to(device) for a in batch]
            x1 = x1.float()
            x2 = x2.float()
            x3 = x3.float()
            dj1 = dj1.float()
            dj2 = dj2.float()
            ### Forward ###
            # First reconstruction
            h1, mu1, logvar1 = nets(x1, dj1[:, dhandler.intervened_on])
            x2_hat = torch.sigmoid(h1)
            # Second reconstruction
            h2, mu2, logvar2 = nets(x1, 
                [dj1[:, dhandler.intervened_on],dj2[:, dhandler.intervened_on]])
            x3_hat = torch.sigmoid(h2)
            ### Losses
            # Reconstruction
            bce_loss1 = var_utils.bce_loss(x2_hat, x2)
            bce_loss2 = var_utils.bce_loss(x3_hat, x3)
            if nets.variational:
                # KL
                kl_loss = var_utils.kl_loss(mu1, logvar1)
                total_loss = config.beta * kl_loss + bce_loss1 + bce_loss2
            else:
                total_loss = bce_loss1 + bce_loss2
            total_loss.backward()
            optim.step()
            ### Logging
            log_text = f'[{epoch}:{i}] loss\t{total_loss.item():.2f} '
            log_text += f'=\tBCE1 {bce_loss1.item():.2f} '
            log_text += f'=\tBCE2 {bce_loss2.item():.2f} '

            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f}'
            logger.info(log_text)
    
    if config.checkpoint:
        checkpoint_dir = os.path.join(config.out_dir,"checkpoint")
        losses = {
            key:val for (key,val) in vars(shared).items() if 'loss' in key}
        ckpt.save_checkpoint( nets, optim, losses=losses, epoch=epochs-1, 
            save_path=checkpoint_dir)

    plt_utils.plot_curves(shared,config,logger,figname=shared.figname)
    return interrupted_training
