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
# @title          :displacementae/supervised/train_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :12/02/2023
# @version        :1.0
# @python_version :3.7.4

import torch
from torch.utils.data import Dataset
from torch.nn import Module
import wandb
import os


from displacementae.networks.variational_utils import bce_loss
from displacementae.utils import checkpoint, sim_utils
from displacementae.utils import plotting_utils as plt_utils

LOSS_LOWEST = 'loss_lowest'
LOSS_LOWEST_EPOCH = 'loss_lowest_epoch'
LOSS_FINAL = 'loss_final'



def setup_optimizer(params, config):
    lr = config.lr
    weight_decay = config.weight_decay
    if config.use_adam:
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def train(dhandler:Dataset, dloader, nets:Module, config, shared, 
          device, logger, mode):
    params = nets.parameters()
    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False
    batch_cnt = 0
    for epoch in range(epochs):
        with torch.no_grad():
            evaluate(dhandler, nets, device, config, shared, logger, mode,
                     epoch, save_fig=True, plot=not config.no_plots)
        
        logger.info(f"Training epoch {epoch}.")
  
        for i, batch in enumerate(dloader):
            if i==config.n_iter:
                break
  
            optim.zero_grad()
 
            x, z = [elem.to(device) for elem in batch]
            # x is of shape [batch_size, channels, height, width]
            # z is of shape [batch_size, n_labels]

            x = x.float()
            z = z.float()

            ### Losses
            if config.net_mode=='encoder':
                z_hat = nets(x)
                loss = (z_hat - z).square().mean()

            elif config.net_mode=='decoder':
                x_hat = torch.sigmoid(nets(z))
                loss = bce_loss(x_hat,x,reduction='none')
                loss = loss.sum()/x.shape[0]

            elif config.net_mode=='grouprepr':
                z_hat = nets(z)
                loss = (z_hat - z).square().mean()
            
            
            ### Step
            loss.backward()
            optim.step()

            ### Logging
            log_text = f'[{epoch}:{i}] loss\t{loss.item():.2f}'
            logger.info(log_text)

            ### WandB Logging
            if config.log_wandb:
                log_dict = {'train/epoch':epoch,'train/loss':loss.item()}
                wandb.log(log_dict,step=batch_cnt,commit=False)
                batch_cnt += 1

    if config.checkpoint:
        checkpoint_dir = os.path.join(config.out_dir, "checkpoint")
        losses = {
            key: val for (key, val) in vars(shared).items() if 'loss' in key}
        checkpoint.save_checkpoint(nets, optim, losses=losses, epoch=epochs-1,
                             save_path=checkpoint_dir)


def evaluate(dhandler:Dataset,
             nets:Module,
             device, config, shared, logger, mode, epoch,
             save_fig=False, plot=False):
    nets.eval()
    if epoch == 0:
        shared.loss = []

    if epoch % config.val_epoch == 0:

        with torch.no_grad():
            batch = dhandler.get_val_batch()
            x, z = [torch.tensor(elem).to(device) for elem in batch]
            # x is of shape [batch_size, channels, height, width]
            # z is of shape [batch_size, n_labels]

            x = x.float()
            z = z.float()

            if config.net_mode=='encoder':
                z_hat = nets(x)
                loss = (z_hat - z).square().mean()

            elif config.net_mode=='decoder':
                x_hat = torch.sigmoid(nets(z))
                loss = bce_loss(x_hat,x,reduction='none')
                loss = loss.sum()/x.shape[0]
            
            elif config.net_mode=='grouprepr':
                z_hat = nets(z)
                loss = (z_hat - z).square().mean()

                
            ### Logging
            logger.info(f'EVALUATION prior to epoch [{epoch}]...')

            log_text = f'[{epoch}] loss\t{loss.item():.2f}'
            logger.info(log_text)

            ### WandB Logging
            if config.log_wandb:
                log_dict = {'val/epoch':epoch,'val/loss':loss.item()}
                wandb.log(log_dict)

            # Save Losses
            shared.loss.append(loss.item())

            shared.summary[LOSS_FINAL] = loss.item()
            if shared.summary[LOSS_LOWEST] == -1 or\
                    loss < shared.summary[LOSS_LOWEST]:
                shared.summary[LOSS_LOWEST] = loss.item()
                shared.summary[LOSS_LOWEST_EPOCH] = epoch

            sim_utils.save_summary_dict(config, shared)

            if epoch % 2*config.val_epoch == 0:
                sim_utils.save_dictionary(shared, config)


    if plot and (epoch % config.plot_epoch == 0):
        with torch.no_grad():
            fig_dir = os.path.join(config.out_dir, 'figures')
            figname = None
            if save_fig:
                figname = os.path.join(fig_dir, f'{epoch}_')
                shared.figname = figname
            
            if config.net_mode == 'decoder':
                plt_utils.plot_supervised_reconstruction(dhandler,nets,config,device,logger,figname)
    nets.train()