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

from typing import List
import numpy as np
import os
import torch
from torch.utils.data import DataLoader

from displacementae.grouprepr.prodrepr.action_lookup import ActionLookup
from displacementae.grouprepr.block_lookup_representation import BlockLookupRepresentation
from displacementae.data.trajs import TrajectoryDataset
import displacementae.networks.network_utils as net_utils
import displacementae.networks.variational_utils as var_utils
import displacementae.utils.plotting_utils as plt_utils
import displacementae.utils.sim_utils as sim_utils
import displacementae.utils.misc as misc
import displacementae.utils.checkpoint as ckpt
from displacementae.networks.multistep_autoencoder import MultistepAutoencoder


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


def evaluate(dhandler: TrajectoryDataset,
             dloader: DataLoader,
             nets: MultistepAutoencoder,
             device, config, shared, logger, mode, epoch,
             save_fig=False, plot=False):
    nets.eval()
    if epoch == 0:
        shared.bce_loss = []
        shared.learned_repr = []
        if config.variational:
            shared.kl_loss = []

    if not epoch % config.val_epoch == 0:
        return

    total_losses, step_losses, bce_losses, kl_losses = [], [], [], []
    for batch in dloader:
        imgs, actions = [torch.as_tensor(elem, dtype=torch.float32, device=device) for elem in batch]
        # imgs is of shape
        # [batch_size, n_steps+1, channels, height, width]
        # actions is of shape [batch_size, n_steps, n_actions]
        x1 = imgs[:, 0]  # initial observed image
        if config.reconstruct_first:
            xi = imgs
        else:
            xi = imgs[:, 1:]  # All other images to predict

        ### Forward ###
        h, mu, logvar = nets.encode(x1)
        h_hat = nets.act(h, actions)
        xi_hat = torch.sigmoid(nets.decode(h_hat))
        if config.latent_loss:
            h_code, _, _ = nets.encode(xi.reshape(-1, *imgs.shape[2:]))
            h_code = h_code.reshape(*xi.shape[:2], h_code.shape[-1])


        ### Losses
        # Reconstruction
        bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
        bce_loss_step = bce_loss_elementwise.sum(dim=[2, 3, 4])
        bce_loss = bce_loss_step.mean(dim=1)
        total_loss = bce_loss

        if nets.variational:
            # KL
            kl_loss = var_utils.kl_loss(mu, logvar)
            total_loss = total_loss + config.beta * kl_loss
        if config.latent_loss:
            latent_loss = (h_code - h_hat).square().mean()
            total_loss = total_loss + config.latent_loss_weight * latent_loss
        if isinstance(nets.grp_morphism, ActionLookup):
            ent_loss = nets.grp_morphism.entanglement_loss()
            total_loss = total_loss + config.grp_loss_weight * ent_loss

        # KL
        if config.variational:
            kl_loss = var_utils.kl_loss(mu, logvar)
            kl_losses.append(kl_loss)
            total_loss = total_loss + config.beta * kl_loss

        bce_losses.append(bce_loss)
        total_losses.append(total_loss)
        step_losses.append(bce_loss_step)

    bce_losses = torch.cat(bce_losses).mean()
    total_loss = torch.cat(total_losses).mean()
    step_loss = torch.cat(step_losses)

    ### Logging
    logger.info(f'EVALUATION prior to epoch [{epoch}]...')
    log_text = f'[{epoch}] loss\t{total_loss.item():.2f}'
    log_text += f'=\tBCE {bce_losses.item():.2f} '
    if nets.variational:
        log_text += f'+\tKL {torch.cat(kl_losses).mean().item():.5f}'
    logger.info(log_text)
    # Save Losses
    np.save(os.path.join(config.out_dir, f'loss_{epoch}'),
            step_loss.cpu().numpy())
    shared.bce_loss.append(step_loss.mean(dim=0).tolist())
    if nets.variational:
        shared.kl_loss.append(kl_loss.item())
    a_in, a = dhandler.get_example_actions()
    #example_R = nets.grp_morphism.get_example_repr(
    #                torch.as_tensor([a_in], dtype=torch.float32, device=device))
    #shared.learned_repr = example_R.tolist()
    if (isinstance(nets.grp_morphism, BlockLookupRepresentation) or
            isinstance(nets.grp_morphism, ActionLookup)):
        reprs = nets.grp_morphism(torch.arange(dhandler.n_actions, device=device))
        np.save(os.path.join(config.out_dir, f'repr_{epoch}'),
                reprs.cpu().numpy())

    shared.actions = [a]

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

    if epoch % 2 * config.val_epoch == 0:
        sim_utils.save_dictionary(shared, config)

    if plot and (epoch % config.plot_epoch == 0):
        fig_dir = os.path.join(config.out_dir, 'figures')
        figname = None
        if save_fig:
            figname = os.path.join(fig_dir, f'{epoch}_')
            shared.figname = figname
        plt_utils.plot_step_recon_loss(step_loss.cpu().numpy(), config, figname)
    #    vary_latents = misc.str_to_ints(config.plot_vary_latents)
    #    plot_latent = misc.str_to_ints(config.plot_manifold_latent)
    #    if len(plot_latent) > 0:
    #        if not isinstance(plot_latent[0], list):
    #            plot_latent = [plot_latent]
    #            vary_latents = [vary_latents]
    #        for i in range(len(vary_latents)):
    #            if config.plot_pca:
    #                plt_utils.plot_manifold_pca(dhandler[1], nets, shared, config,
    #                                            device, logger, mode, epoch,
    #                                            vary_latents=vary_latents[i],
    #                                            figname=figname)
    #            else:
    #                plt_utils.plot_manifold(dhandler[1], nets, shared, config,
    #                                        device, logger, mode, epoch,
    #                                        vary_latents=vary_latents[i],
    #                                        plot_latent=plot_latent[i],
    #                                        figname=figname)
    nets.train()


def train(dhandler: List[TrajectoryDataset],
          dloader: List[DataLoader],
          nets: MultistepAutoencoder, config, shared, device, logger, mode):
    params = nets.parameters()
    optim = setup_optimizer(params, config)
    interrupted_training = False

    for epoch in range(config.epochs):
        with torch.no_grad():
            evaluate(dhandler[1], dloader[1], nets, device, config, shared, logger, mode,
                     epoch, save_fig=True, plot=not config.no_plots)

        logger.info(f"Training epoch {epoch}.")
        nets.grp_morphism.end_iteration()
        nets.train()
        for i, batch in enumerate(dloader[0]):
            optim.zero_grad()

            imgs, actions = [torch.as_tensor(elem, dtype=torch.float32, device=device) for elem in batch]

            x1 = imgs[:, 0]  # initial observed image
            if config.reconstruct_first:
                xi = imgs
            else:
                xi = imgs[:, 1:]  # All other images to predict


            ### Forward ###
            h, mu, logvar = nets.encode(x1)
            h_hat = nets.act(h, actions)
            xi_hat = torch.sigmoid(nets.decode(h_hat))
            if config.latent_loss:
                h_code, _, _ = nets.encode(xi.reshape(-1, *imgs.shape[2:]))
                h_code = h_code.reshape(*xi.shape[:2], h_code.shape[-1])

            ### Losses
            # Reconstruction
            bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            bce_loss = bce_loss_elementwise.sum(dim=[2, 3, 4]).mean()
            total_loss = bce_loss

            if nets.variational:
                # KL
                kl_loss = var_utils.kl_loss(mu, logvar)
                total_loss = total_loss + config.beta * kl_loss
            if config.latent_loss:
                latent_loss = (h_code - h_hat).square().mean()
                total_loss = total_loss + config.latent_loss_weight * latent_loss

            if isinstance(nets.grp_morphism, ActionLookup):
                ent_loss = nets.grp_morphism.entanglement_loss()
                total_loss = total_loss + config.grp_loss_weight * ent_loss

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
            log_text += f'=\tBCE {bce_loss.item():.2f} '

            if nets.variational:
                log_text += f'+\tKL {kl_loss.item():.5f}'
            if config.latent_loss:
                log_text += f'\tLL {latent_loss.item():.5f} '
            if isinstance(nets.grp_morphism, ActionLookup):
                log_text += f'\tEL {ent_loss.item():.5f} '
            log_text += f'\tG-Norm {total_norm.item():.2f}/{decoder_norm.item():.2f}/{act_norm.item():.2f} '
            logger.info(log_text)

    if config.checkpoint:
        checkpoint_dir = os.path.join(config.out_dir, "checkpoint")
        losses = {
            key: val for (key, val) in vars(shared).items() if 'loss' in key}
        ckpt.save_checkpoint(nets, optim, losses=losses, epoch=config.epochs-1,
                             save_path=checkpoint_dir)

    plt_utils.plot_curves(shared, config, logger, figname=shared.figname)
    return interrupted_training
