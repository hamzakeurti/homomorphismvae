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
import wandb
import numpy as np

import data.data_utils as data_utils
import data.transition_dataset as trns_data
import networks.network_utils as net_utils
import networks.variational_utils as var_utils
import utils.plotting_utils as plt_utils
import utils.sim_utils as sim_utils
import utils.misc as misc
import utils.checkpoint as ckpt
import networks.multistep_autoencoder as ms_ae
from utils.scheduler import setup_scheduler


BCE_LOWEST = 'bce_lowest'
KL_HIGHEST = 'kl_highest'
LOSS_LOWEST = 'loss_lowest'
LOSS_LOWEST_EPOCH = 'loss_lowest_epoch'
BCE_FINAL = 'bce_final'
KL_FINAL = 'kl_final'
LOSS_FINAL = 'loss_final'


def setup_optimizer(params, config):
    lr = config.lr
    weight_decay = config.weight_decay
    if config.use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def evaluate(dhandler:trns_data.TransitionDataset, 
             nets:ms_ae.MultistepAutoencoder, 
             device, config, shared, logger, mode, epoch,
             save_fig=False, plot=False):
    nets.eval()
    if epoch == 0:
        shared.bce_loss = []
        shared.learned_repr = []
        if config.variational:
            shared.kl_loss = []
        if config.grp_loss_on:
            shared.grp_loss = []

        
    if epoch % config.val_epoch == 0:

        with torch.no_grad():
            batch = dhandler.get_val_batch()
            imgs, latents, dj = [torch.from_numpy(elem).to(device) 
                                 for elem in batch]
            # imgs is of shape 
            # [batch_size, n_steps+1, channels, height, width]
            # dj is of shape [batch_size, n_steps, n_actions]

            imgs = imgs.float()
            dj = dj.float()

            x1 = imgs[:,0] # initial observed image
            xi = imgs[:,1:] # All other images to predict
            ### Forward ###
            h, mu, logvar = nets(x1, dj)
            xi_hat = torch.sigmoid(h)
            ### Losses
            # Reconstruction
            bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            bce_loss_per_image =\
                bce_loss_elementwise.sum(dim=[0,2,3,4])/x1.shape[0]
            bce_loss = bce_loss_per_image.sum()/config.n_steps
            total_loss = bce_loss
            # KL
            if config.variational:
                kl_loss = var_utils.kl_loss(mu, logvar)
                total_loss += config.beta * kl_loss
            if config.grp_loss_on:
                grp_loss = nets.grp_morphism.representation_loss(dj)
                total_loss += grp_loss * config.grp_loss_weight
                shared.grp_loss.append(grp_loss.item())
            # Get representation matrices for typical actions
            a_in, a = dhandler.get_example_actions()
            example_R = nets.grp_morphism.get_example_repr(
                            torch.from_numpy(a_in).float().to(device))

            ### Logging
            logger.info(f'EVALUATION prior to epoch [{epoch}]...') 
            log_text = f'[{epoch}] loss\t{total_loss.item():.2f}'
            log_text += f'=\tBCE {bce_loss.item():.2f} '
            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f} '
            if config.grp_loss_on:
                log_text += f'=\tGRP {grp_loss.item():.2f}'
            logger.info(log_text)


            ### WandB Logging
            if config.log_wandb:
                log_dict = {'val/epoch':epoch,'val/total_loss':total_loss.item(),
                            'val/bce_loss':bce_loss.item()}
                if nets.variational:
                    log_dict['val/kl_loss'] = kl_loss.item()
                wandb.log(log_dict)
            if (epoch % config.plot_epoch == 0):
                # log_dict['val/learned_repr']=example_R.tolist()
                fig_dir = os.path.join(config.out_dir, 'figures')
                figname=os.path.join(fig_dir,'learned_repr_')
                plt_utils.plot_matrix(example_R, a, config,
                                      logger, figname=figname)
        
            # Save Losses
            shared.bce_loss.append(bce_loss_per_image.tolist())
            if nets.variational:
                shared.kl_loss.append(kl_loss.item())
            if config.grp_loss_on:
                shared.grp_loss.append(grp_loss.item())

            shared.learned_repr = example_R.tolist()
            shared.actions = a.tolist()

            shared.summary[LOSS_FINAL] = total_loss.item()
            if shared.summary[LOSS_LOWEST] == -1 or\
                    total_loss < shared.summary[LOSS_LOWEST]:
                shared.summary[LOSS_LOWEST] = total_loss.item()
                shared.summary[LOSS_LOWEST_EPOCH] = epoch
            
            if config.variational:
                shared.summary[KL_FINAL] = kl_loss.item()
                if shared.summary[KL_HIGHEST] == -1 or\
                        kl_loss > shared.summary[KL_HIGHEST]:
                    shared.summary[KL_HIGHEST] = kl_loss.item()
                    # shared.summary[LOSS_LOWEST_EPOCH] = epoch
                shared.summary[BCE_FINAL] = bce_loss.item()
                if shared.summary[BCE_LOWEST] == -1 or\
                        bce_loss < shared.summary[BCE_LOWEST]:
                    shared.summary[BCE_LOWEST] = bce_loss.item()

            sim_utils.save_summary_dict(config, shared)

            if epoch % 2*config.val_epoch == 0:
                sim_utils.save_dictionary(shared,config)
            
    if plot and (epoch % config.plot_epoch == 0):
        with torch.no_grad():
            fig_dir = os.path.join(config.out_dir, 'figures')
            figname = None
            if save_fig:
                figname = os.path.join(fig_dir, f'{epoch}_')
                shared.figname=figname
            plt_utils.plot_n_step_reconstruction(dhandler, nets, config, 
                                                 device, logger,epoch,figname)
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

def train(dhandler, dloader, nets:ms_ae.MultistepAutoencoder, config, shared, 
          device, logger, mode):
    params = nets.parameters()
    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False
    scheduler = setup_scheduler(
                    config,
                    group1=[nets.encoder,nets.grp_morphism,nets.decoder], 
                    group2 = [nets.encoder,nets.decoder])
    batch_cnt = 0
    for epoch in range(epochs):
        with torch.no_grad():
            evaluate(dhandler, nets, device, config, shared, logger, mode,
                     epoch, save_fig=True, plot=not config.no_plots)

        logger.info(f"Training epoch {epoch}.")
        # scheduler.toggle_train(
        #     [nets.encoder,nets.grp_morphism,nets.decoder], 
        #     [nets.encoder,nets.decoder],
        #     epoch)
        scheduler.toggle_train()
        

        for i, batch in enumerate(dloader):
            optim.zero_grad()

            imgs, latents, dj = [elem.to(device) for elem in batch]
            # imgs is of shape [batch_size, n_steps+1, channels, height, width]
            # dj is of shape [batch_size, n_steps, n_actions]

            imgs = imgs.float()
            dj = dj.float()

            x1 = imgs[:,0] # initial observed image
            xi = imgs[:,1:] # All other images to predict
            ### Forward ###
            h, mu, logvar = nets(x1, dj)
            xi_hat = torch.sigmoid(h)
            ### Losses
            # Reconstruction
            bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            bce_loss_per_image =\
                bce_loss_elementwise.sum(dim=[0,2,3,4])/x1.shape[0]
            bce_loss = bce_loss_per_image.sum()/config.n_steps
            total_loss = bce_loss
            if nets.variational:
                # KL
                kl_loss = var_utils.kl_loss(mu, logvar)
                total_loss += config.beta * kl_loss
            if config.grp_loss_on:
                grp_loss = nets.grp_morphism.representation_loss(dj)
                total_loss += grp_loss * config.grp_loss_weight
            total_loss.backward()
            optim.step()
            # Clear the stored matrices so they are regenerated at next 
            # iteration. Only does something if representation is PROD_ROTS.
            nets.grp_morphism.end_iteration()

            ### Logging
            log_text = f'[{epoch}:{i}] loss\t{total_loss.item():.2f} '
            log_text += f'=\tBCE {bce_loss.item():.2f} '

            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f}'
            if config.grp_loss_on:
                log_text += f'=\tGRP {grp_loss.item():.2f}'
            logger.info(log_text)
            
            ### WandB Logging
            if config.log_wandb:
                log_dict = {'train/epoch':epoch,'train/total_loss':total_loss.item(),
                            'train/bce_loss':bce_loss.item()}
                if nets.variational:
                    log_dict['train/kl_loss'] = kl_loss.item()
                if config.grp_loss_on:
                    log_dict['train/grp_loss'] = grp_loss.item()
                wandb.log(log_dict,step=batch_cnt,commit=False)
                batch_cnt += 1
    
    if config.checkpoint:
        checkpoint_dir = os.path.join(config.out_dir,"checkpoint")
        losses = {
            key:val for (key,val) in vars(shared).items() if 'loss' in key}
        ckpt.save_checkpoint( nets, optim, losses=losses, epoch=epochs-1, 
            save_path=checkpoint_dir)

    plt_utils.plot_curves(shared,config,logger,figname=shared.figname)
    return interrupted_training
