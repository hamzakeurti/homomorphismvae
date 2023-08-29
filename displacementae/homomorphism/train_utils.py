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

from displacementae.data.transition_dataset import TransitionDataset
from displacementae.networks.multistep_autoencoder import MultistepAutoencoder
import displacementae.data.data_utils as data_utils
import displacementae.data.transition_dataset as trns_data
import displacementae.networks.network_utils as net_utils
import displacementae.networks.variational_utils as var_utils
import displacementae.utils.plotting_utils as plt_utils
import displacementae.utils.sim_utils as sim_utils
import displacementae.utils.misc as misc
import displacementae.utils.checkpoint as ckpt
import displacementae.networks.multistep_autoencoder as ms_ae
from displacementae.utils.scheduler import setup_scheduler


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
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, amsgrad=True)
    else:
        optimizer = torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    return optimizer


def evaluate(dhandler:TransitionDataset,
             nets:MultistepAutoencoder,
             device, config, shared, logger, mode, epoch,
             save_fig=False, plot=False, plot_reconstruction:bool=False, 
             plot_manifold:bool=False, plot_matrices:bool=False):

    nets.eval()
    if epoch == 0:
        shared.bce_loss = []
        shared.learned_repr = []
        if config.rollouts:
            shared.rollout_errors = []
        if config.variational:
            shared.kl_loss = []
        if nets.grp_morphism.repr_loss_on:
            shared.grp_loss = []

    if epoch % config.val_epoch == 0:

        with torch.no_grad():
            # Evaluation on left out data
            batch = dhandler.get_val_batch()
            imgs, _, dj = batch
            imgs, dj = [torch.from_numpy(elem).to(device)
                                 for elem in [imgs, dj]]
            # imgs is of shape
            # [batch_size, n_steps+1, channels, height, width]
            # dj is of shape [batch_size, n_steps, n_actions]

            imgs = imgs.float()
            dj = dj.float()
            with torch.no_grad():
                matrices = torch.cat([nets.grp_morphism(dj[:, i]) for i in range(dj.shape[1])])
                u, s, v = torch.svd(matrices)
                #print(s)

            x1 = imgs[:,0] # initial observed image
            # All other images to predict
            if config.reconstruct_first:
                xi = imgs
            elif config.reconstruct_first_only:
                xi = imgs[:,0]
            else:    
                xi = imgs[:,1:]
          
            # =========================
            # ### Forward ###
            # h, latent, latent_hat, mu, logvar = nets(x1, dj)
            # xi_hat = torch.sigmoid(h)
            # =========================

            ### Forward
            h, mu, logvar = nets.encode(x1)
            h_hat = nets.act(h, dj)
            xi_hat = torch.sigmoid(nets.decode(h_hat))
            if config.latent_loss:
                # h_code, _, _ = nets.encode(xi.reshape(-1, *imgs.shape[2:]))
                # h_code = h_code.reshape(*xi.shape[:2], h_code.shape[-1])

                h_code, _, _ = nets.encode(imgs.reshape(-1, *imgs.shape[2:]))
                h_code = h_code.reshape(*imgs.shape[:2], h_code.shape[-1])


            ### Losses
            # Reconstruction
            if config.reconstruct_first:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            elif config.reconstruct_first_only:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat[:,0], imgs[:,0], 'none')
                bce_loss_elementwise = bce_loss_elementwise[:,None]
            else:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat[:,1:], imgs[:,1:], 'none')
            n_img = bce_loss_elementwise.shape[1]
            bce_loss_per_image =\
                bce_loss_elementwise.sum(dim=[0,2,3,4])/x1.shape[0]
            bce_loss = bce_loss_per_image.sum()/n_img
            total_loss = bce_loss
            # KL
            if config.variational:
                kl_loss = var_utils.kl_loss(mu, logvar)
                total_loss += config.beta * kl_loss
            # Latent Loss
            if config.latent_loss:
                latent_loss = (h_code[:,1:] - h_hat[:,1:]).square().mean()
                total_loss += config.latent_loss_weight * latent_loss
            # Grp Loss
            if nets.grp_morphism.repr_loss_on:
                grp_loss = nets.grp_morphism.representation_loss(dj)
                total_loss += config.grp_loss_weight * grp_loss


            ### Logging
            logger.info(f'EVALUATION prior to epoch [{epoch}]...')
            log_text = f'[{epoch}] loss\t{total_loss.item():.2f}'
            log_text += f'=\tBCE {bce_loss.item():.2f} '
            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f} '
            if config.latent_loss:
                log_text += f'\tLL {latent_loss.item():.5f} '
            if nets.grp_morphism.repr_loss_on:
                log_text += f'\tGL {grp_loss.item():.5f} '

            logger.info(log_text)
            
            ### WandB Logging
            if config.log_wandb:
                log_dict = {'val/epoch':epoch,'val/total_loss':total_loss.item(),
                            'val/bce_loss':bce_loss.item()}
                if nets.variational:
                    log_dict['val/kl_loss'] = kl_loss.item()
                if config.latent_loss:
                    log_dict['val/ll_loss'] = latent_loss.item()
                if nets.grp_morphism.repr_loss_on:
                    log_dict['val/gl_loss'] = grp_loss.item()
                wandb.log(log_dict)

            if config.plot_matrices and (epoch % config.plot_epoch == 0):
            # Get representation matrices for typical actions
                a_in, a = dhandler.get_example_actions()
                example_R = nets.grp_morphism.get_example_repr(
                                torch.from_numpy(a_in).float().to(device))

                shared.actions = a.tolist()
                shared.learned_repr = example_R.tolist()        
                # log_dict['val/learned_repr']=example_R.tolist()
                fig_dir = os.path.join(config.out_dir, 'figures')
                figname=os.path.join(fig_dir,'learned_repr_')
                plt_utils.plot_matrix(example_R, a, config,
                                      logger, figname=figname)
            # Save Losses
            shared.bce_loss.append(bce_loss_per_image.tolist())
            if nets.variational:
                shared.kl_loss.append(kl_loss.item())
            if nets.grp_morphism.repr_loss_on:
                shared.grp_loss.append(grp_loss.item())

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
                shared.figname = figname
            if config.plot_reconstruction:
                dhandler.plot_n_step_reconstruction(nets, config,
                                                 device, logger, figname)
            
            if config.plot_manifold:
                if config.plot_pca:
                    dhandler.plot_manifold_pca(nets, shared, config,
                                               device, logger, mode, 
                                               epoch, fig_dir)
                else:
                    dhandler.plot_manifold(nets, shared, config, 
                                           device, logger, mode, epoch, fig_dir)
    
    if config.rollouts:
        evaluate_rollouts(dhandler, nets, device, config, shared, logger, mode, 
                          epoch, plot_rollouts=config.plot_rollouts)
    
    nets.train()


def evaluate_rollouts(dhandler:TransitionDataset, nets:MultistepAutoencoder,
                      device, config, shared, logger, mode, epoch, 
                      plot_rollouts=False):
    nets.eval()
    
    if epoch % config.val_epoch == 0:
        with torch.no_grad():
            errors = []
            n = 0
            for X, a in dhandler.get_rollouts():
                X = torch.FloatTensor(X).to(device)
                a = torch.FloatTensor(a).to(device)
                X_hat, _,_,_,_ = nets(X[:,0], a) 
                X_hat = torch.nan_to_num(X_hat,nan=0.0)
                X_hat = torch.sigmoid(X_hat)
                bce_loss_elementwise = var_utils.bce_loss(X_hat, X, 'none')
                bce_loss_per_image =\
                    bce_loss_elementwise.sum(dim=[0,2,3,4])/X.shape[0]
                errors.append(bce_loss_per_image.cpu().numpy())
                n += X.shape[0]
            errors = np.vstack(errors)
            errors = errors.sum(axis=0)/n
            avg_error = errors.mean()
            i_pow2 = np.power(2, np.arange(2, np.log2(len(errors)),1)).astype(int)
            shared.rollout_errors.append(errors[i_pow2].tolist())

            logger.info(f'[{epoch}] EVALUATION rollouts over {a.shape[1]} steps')
            log_text = f'[{epoch}] avg bce loss\t{avg_error:.2f}'
            logger.info(log_text)

            ### WandB Logging
            if config.log_wandb:
                log_dict = {'val/rollouts/epoch':epoch,
                            f'val/rollouts/avg_error_{errors.shape[-1]}_steps':avg_error,}
                for p in i_pow2:
                    p = int(p)
                    log_dict[f'val/rollouts/error_step_{p}'] = errors[p]
        
                wandb.log(log_dict)

        
    if plot_rollouts and (epoch % config.plot_epoch == 0):
        dhandler.plot_rollout_reconstruction(
                        nets=nets, config=config, 
                        device=device, logger=logger)
    nets.train()


def train(dhandler:TransitionDataset, dloader, nets:MultistepAutoencoder, 
          config, shared, device, logger, mode):
    params = nets.parameters()
    optim = setup_optimizer(params, config)
    epochs = config.epochs
    interrupted_training = False
    scheduler = setup_scheduler(
                    config,
                    group1=[nets.encoder, nets.grp_morphism, nets.decoder],
                    group2=[nets.encoder, nets.decoder])
    batch_cnt = 0
    for epoch in range(epochs):
        if epoch % config.resample_every == (config.resample_every -1):
            dhandler.resample_data()
        with torch.no_grad():
            evaluate(dhandler, nets, device, config, shared, logger, mode,
                     epoch, save_fig=True, plot=not config.no_plots)

        logger.info(f"Training epoch {epoch}.")
        # scheduler.toggle_train(
        #     [nets.encoder,nets.grp_morphism,nets.decoder],
        #     [nets.encoder,nets.decoder],
        #     epoch)
        #scheduler.toggle_train()

        for i, batch in enumerate(dloader):
            if i==config.n_iter:
                break
            optim.zero_grad()
            imgs, _, dj = batch 
            imgs, dj = [elem.to(device) for elem in (imgs, dj)]
            # imgs is of shape [batch_size, n_steps+1, channels, height, width]
            # dj is of shape [batch_size, n_steps, n_actions]

            imgs = imgs.float()
            dj = dj.float()

            x1 = imgs[:,0] # initial observed image
            # All other images to predict
            if config.reconstruct_first:
                xi = imgs
            else:
                xi = imgs[:,1:]
            ### Forward ###
            # h, latent, latent_hat, mu, logvar = nets(x1, dj)
            # latent_nstep, _, _ = nets.encode(xi.reshape(-1, *imgs.shape[2:]))
            # latent_nstep = latent_nstep.reshape(*xi.shape[:2], latent_nstep.shape[-1])
            # x1_hat = torch.sigmoid(nets.decoder(latent))
            # xi_hat = torch.sigmoid(h)

            ### Forward
            h, mu, logvar = nets.encode(x1)
            h_hat = nets.act(h, dj)
            xi_hat = torch.sigmoid(nets.decode(h_hat))
            if config.latent_loss:
                # h_code, _, _ = nets.encode(xi.reshape(-1, *imgs.shape[2:]))
                # h_code = h_code.reshape(*xi.shape[:2], h_code.shape[-1])

                h_code, _, _ = nets.encode(imgs.reshape(-1, *imgs.shape[2:]))
                h_code = h_code.reshape(*imgs.shape[:2], h_code.shape[-1])


            ### Losses
            # Reconstruction
            if config.reconstruct_first:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            elif config.reconstruct_first_only:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat[:,0], imgs[:,0], 'none')
                bce_loss_elementwise = bce_loss_elementwise[:,None]
            else:
                bce_loss_elementwise = var_utils.bce_loss(xi_hat[:,1:], imgs[:,1:], 'none')
            n_img = bce_loss_elementwise.shape[1]
            bce_loss_per_image =\
                bce_loss_elementwise.sum(dim=[0,2,3,4])/x1.shape[0]
            bce_loss = bce_loss_per_image.sum()/n_img
            total_loss = bce_loss
            # KL
            if config.variational:
                kl_loss = var_utils.kl_loss(mu, logvar)
                total_loss += config.beta * kl_loss
            # Latent Loss
            if config.latent_loss:
                latent_loss = (h_code[:,1:] - h_hat[:,1:]).square().mean()
                total_loss += config.latent_loss_weight * latent_loss
            # Grp Loss
            if nets.grp_morphism.repr_loss_on:
                grp_loss = nets.grp_morphism.representation_loss(dj)
                total_loss += config.grp_loss_weight * grp_loss

            total_loss.backward()
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in nets.parameters() if p.grad is not None]))
            decoder_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in nets.decoder.parameters() if p.grad is not None]))
            act_norm = torch.norm(torch.stack([torch.norm(p.grad.detach()).to(device) for p in nets.grp_morphism.parameters() if p.grad is not None]))
            optim.step()
            # Clear the stored matrices so they are regenerated at next
            # iteration.
            nets.grp_morphism.end_iteration()

            ### Logging
            log_text = f'[{epoch}:{i}] loss\t{total_loss.item():.2f} '
            log_text += f'=\tBCE {bce_loss.item():.5f} '

            if config.variational:
                log_text += f'+\tKL {kl_loss.item():.5f}'
            if config.latent_loss:
                log_text += f'\tLL {latent_loss.item():.5f} '
            if nets.grp_morphism.repr_loss_on:
                log_text += f'\tGL {grp_loss.item():.5f} '
            
            log_text += f'\tTotal {total_norm.item():.2f}/{decoder_norm.item():.2f}/{act_norm.item():.2f} '
            logger.info(log_text)
            
            ### WandB Logging
            if config.log_wandb:
                log_dict = {'train/epoch':epoch,'train/total_loss':total_loss.item(),
                            'train/bce_loss':bce_loss.item()}
                if config.variational:
                    log_dict['train/kl_loss'] = kl_loss.item()
                if config.latent_loss:
                    log_dict['train/ll_loss'] = latent_loss.item()                
                if nets.grp_morphism.repr_loss_on:
                    log_dict['train/gl_loss'] = grp_loss.item()
                wandb.log(log_dict,step=batch_cnt,commit=False)
                batch_cnt += 1
        
        if config.checkpoint and (epoch > 0) and (epoch % config.checkpoint_every == 0):
            checkpoint_dir = os.path.join(config.out_dir, "checkpoint")
            losses = {
                key: val for (key, val) in vars(shared).items() if 'loss' in key}
            ckpt.save_checkpoint(nets, optim, losses=losses, epoch=epoch,
                             save_path=checkpoint_dir)
    
    if config.checkpoint:
        checkpoint_dir = os.path.join(config.out_dir, "checkpoint")
        losses = {
            key: val for (key, val) in vars(shared).items() if 'loss' in key}
        ckpt.save_checkpoint(nets, optim, losses=losses, epoch=epochs-1,
                             save_path=checkpoint_dir)

    plt_utils.plot_curves(shared, config, logger, figname=shared.figname)
    return interrupted_training
