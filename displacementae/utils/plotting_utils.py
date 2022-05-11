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
# @title          :displacementae/utils/plotting_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :22/11/2021
# @version        :1.0
# @python_version :3.7.4

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from argparse import Namespace
import wandb

import networks.autoencoder_prodrep as aeprod
import utils.misc as misc
import utils.data_utils as udutils
from sklearn.random_projection import GaussianRandomProjection
from scipy.stats import sem

_DEFAULT_PLOT_CONFIG = [12, 5, 8] # fontsize, linewidth, markersize

# Simple 2D regression grid.
_TWO_D_MISC = Namespace()
_TWO_D_MISC.x_range = [-5, 5]
_TWO_D_MISC.y_range = [-5, 5]
_TWO_D_MISC.x_range_medium = [-2, 2]
_TWO_D_MISC.y_range_medium = [-2, 2]
_TWO_D_MISC.x_range_narrow = [-0.3, 0.3]
_TWO_D_MISC.y_range_narrow = [-0.3, 0.3]
MARKERS = np.array(
        ["o","^", ">","v","<","1","2","3","4","8","s","p","*","+","x","d"])


def plot_reconstruction(dhandler, nets, config, device, logger, epoch,
                        figname):
    if config.plot_on_black:
        plt.style.use('dark_background')
    
        
    imgs, latents, dj = dhandler.get_val_batch()
    X1 = torch.FloatTensor(imgs[:,0]).to(device)
    if config.intervene:
        X2 = torch.FloatTensor(imgs[:,1]).to(device)
    else:
        X2 = X1.clone()
    dj = torch.FloatTensor(dj).to(device).squeeze()
    if config.intervene:
        h, _, _, mu, logvar = nets(X1, dj)
    else:
        h, _, _, mu, logvar = nets(X1, None)
    X2_hat = torch.sigmoid(h)
    nrows = 7
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5, 8))
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
    for row in range(nrows):
        axes[row, 0].imshow(X1[row, 0].cpu().numpy(), **kwargs)
        axes[row, 1].imshow(X2[row, 0].cpu().numpy(), **kwargs)
        axes[row, 2].imshow(X2_hat[row, 0].cpu().numpy(), **kwargs)
        if config.plot_on_black:
            for i in range(3):
                axes[row, i].axes.xaxis.set_visible(False)
                axes[row, i].axes.yaxis.set_visible(False)
        else:
            for i in range(3):
                axes[row, i].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.1)

    if figname is not None:
        figname += 'reconstructions.pdf'
        plt.savefig(figname,bbox_inches='tight')
        logger.info(f'Figure saved {figname}')
    if config.log_wandb:
        wandb.log({'plot/reconstruction':wandb.Image(plt)})
    plt.close(fig)


def plot_n_step_reconstruction(dhandler, nets, config, device, logger, figname):
    if config.plot_on_black:
        plt.style.use('dark_background')

    n_steps = config.n_steps

    imgs, latents, dj = dhandler.get_val_batch()
    X1 = torch.FloatTensor(imgs[:,0]).to(device)
    if config.reconstruct_first:
        Xi = torch.FloatTensor(imgs).to(device)
    else:
        Xi = torch.FloatTensor(imgs[:,1:]).to(device)

    dj = torch.FloatTensor(dj).to(device)

    h, _, _, mu, logvar = nets(X1, dj)
    Xi_hat = torch.sigmoid(h)

    nrows = 7
    if config.reconstruct_first:
        ncols = 2 + 2*n_steps
    else:
        ncols = 1 + 2*n_steps

    unit_length = 1.5

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * unit_length,
                                      nrows * unit_length))
    kwargs = {'vmin': 0, 'vmax': 1, 'cmap': 'gray'}
    for row in range(nrows):
        axes[row,0].imshow(X1[row,0].cpu().numpy(),**kwargs)
        if config.reconstruct_first:
            axes[row,1].imshow(Xi_hat[row,0,0].cpu().numpy(),**kwargs)
            s = 2
        else:
            s = 1
        for i in range(n_steps):
            axes[row,2*i+s].imshow(Xi[row,i,0].cpu().numpy(),**kwargs)
            axes[row,2*i+s+1].imshow(Xi_hat[row,i,0].cpu().numpy(),**kwargs)
        if config.plot_on_black:
            for j in range(ncols):
                axes[row, j].axes.xaxis.set_visible(False)
                axes[row, j].axes.yaxis.set_visible(False)
        else:
            for j in range(ncols):
                axes[row, j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.1)

    if figname is not None:
        figname += 'reconstructions.pdf'
        plt.savefig(figname, bbox_inches='tight')
        logger.info(f'Figure saved {figname}')
    if config.log_wandb:
        wandb.log({'plot/reconstructions':wandb.Image(plt)})
    plt.close(fig)


def plot_manifold(dhandler, nets, shared, config, device, logger, mode,
                  epoch, vary_latents=[3], plot_latent=[0,1], figname=None):
    """
    Produces colored scatter plot of the latent representation of 
    the different positions in the joint space.

    A 1D or 2D grid of joint positions are generated, 
    corresponding images are forwarded through the encoder.
    Resulting latent units are 
    """
    n_repr_units = nets.n_repr_units
    if max(plot_latent) >= n_repr_units:
        raise ValueError(
            "Requested plotting a representational unit which index: "+
            f"{max(plot_latent)} is too large for the "+
            f"number of representational units: {n_repr_units}")
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.plot_on_black:
        plt.style.use('dark_background')

    indices = dhandler.get_indices_vary_latents(vary_latents)
    latents = dhandler.latents[indices][:,vary_latents]
    batch_size = config.batch_size
    n_batches = len(indices) // batch_size + 1
    
    results = []

    for i in range(n_batches):
        batch_indices = indices[ i * batch_size : (i+1) * batch_size]
        images = dhandler.images[batch_indices]
        X = torch.FloatTensor(images).to(device)
        with torch.no_grad():
            h, mu, logvar = nets.encode(X)
            h = nets.normalize_representation(h)
            results.append(h[:,plot_latent].cpu().numpy())
    results = np.vstack(results).squeeze()
    
    if config.plot_on_black:
        kwargs={'cmap':'summer'}
    else:
        kwargs={}

    kwargs['alpha']=0.8
    kwargs['edgecolors']='none'

    for i in range(len(vary_latents)):
        latent = vary_latents[i]
        latent_name = dhandler.get_latent_name(latent)
        fig, ax = plt.subplots(figsize=(8,7))
        if len(plot_latent) == 1:
            f = ax.scatter(x=latents[:,i], y=results)
            ax.set_xlabel(f'true label {latent}', fontsize=ts)
            ax.set_ylabel(f'latent {plot_latent[0]}', fontsize=ts)
        if len(plot_latent) == 2:
            f = ax.scatter(x=results[:,0], y=results[:,1], c=latents[:,i],
                            **kwargs)
            ax.set_xlabel(f'latent {plot_latent[0]}', fontsize=ts)
            ax.set_ylabel(f'latent {plot_latent[1]}', fontsize=ts)
            dx = np.abs(results).max()
            #if config.spherical:
            #    ax.set_xlim(_TWO_D_MISC.x_range_medium)
            #    ax.set_ylim(_TWO_D_MISC.y_range_medium)
            #elif dx <= 0.3:
            #    ax.set_xlim(_TWO_D_MISC.x_range_narrow)
            #    ax.set_ylim(_TWO_D_MISC.y_range_narrow)
            #else:
            #    ax.set_xlim(_TWO_D_MISC.x_range)
            #    ax.set_ylim(_TWO_D_MISC.y_range)
            plt.colorbar(f)
        ax.set_title('Manifold latent for latent: ' + latent_name)
        if figname is not None:
            figname1 = figname + 'repr_units=' + misc.ints_to_str(plot_latent) 
            figname1 += '_varied='+ misc.ints_to_str(vary_latents)
            figname1 += '_clr='+ misc.ints_to_str(latent) + '.pdf'
            plt.savefig(figname1)
            logger.info(f'Figure saved {figname1}')
        if config.log_wandb:
            wandb.log({f'plot/manifold_repr{misc.ints_to_str(plot_latent)}'+
                       f'_varied{misc.ints_to_str(vary_latents)}_col{i}':\
                                                        wandb.Image(plt)})
        plt.close(fig)

def plot_manifold_pca(dhandler, nets, shared, config, device, logger, mode,
                epoch, vary_latents=[3], figname=None):
    """
    Produces colored scatter plot of the latent representation of 
    the different positions in the joint space.

    A 1D or 2D grid of joint positions are generated, 
    corresponding images are forwarded through the encoder.
    Resulting latent units are 
    """
    n_repr_units = nets.n_repr_units
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if config.plot_on_black:
        plt.style.use('dark_background')

    indices = dhandler.get_indices_vary_latents(vary_latents)
    latents = dhandler.latents[indices][:,vary_latents]
    batch_size = config.batch_size
    n_batches = len(indices) // batch_size + 1
    
    results = []

    for i in range(n_batches):
        batch_indices = indices[ i * batch_size : (i+1) * batch_size]
        images = dhandler.images[batch_indices]
        X = torch.FloatTensor(images).to(device)
        with torch.no_grad():
            h, mu, logvar = nets.encode(X)
            h = nets.normalize_representation(h)
            results.append(h[:,:].cpu().numpy())
    results = np.vstack(results).squeeze()

    # PCA Projection
    pca = GaussianRandomProjection(n_components=2)
    latent2d = pca.fit_transform(results)

    # latent2d/= np.linalg.norm(latent2d,)

    if config.plot_on_black:
        kwargs={'cmap':'summer'}
    else:
        kwargs={}

    kwargs['alpha'] = 0.5
    kwargs['edgecolors']='none'


    for i in range(len(vary_latents)):
        if len(vary_latents) > 1:
            kwargs['vmin'] = min(latents[:,i])
            kwargs['vmax'] = max(latents[:,i])
            
        latent = vary_latents[i]
        latent_name = dhandler.get_latent_name(latent)
        fig, ax = plt.subplots(figsize=(8,7))

        # f = ax.scatter(x=latent2d[:,0], y=latent2d[:,1], c=latents[:,i],
                        # **kwargs)
        for x,y,c,m in zip(
                latent2d[:,0],latent2d[:,1],latents[:,i],latents[:,(i+1)%2]):
            f = ax.scatter(x=x, y=y, c=c,marker=MARKERS[m%len(MARKERS)],
                        **kwargs)

        ax.set_xlabel(f'latent component 0', fontsize=ts)
        ax.set_ylabel(f'latent component 1', fontsize=ts)
        dx = np.abs(latent2d).max()
        #if config.spherical:
        #    ax.set_xlim(_TWO_D_MISC.x_range_medium)
        #    ax.set_ylim(_TWO_D_MISC.y_range_medium)
        #elif dx <= 0.3:
        #    ax.set_xlim(_TWO_D_MISC.x_range_narrow)
        #    ax.set_ylim(_TWO_D_MISC.y_range_narrow)
        #else:
        #    ax.set_xlim(_TWO_D_MISC.x_range)
        #    ax.set_ylim(_TWO_D_MISC.y_range)
        
        plt.colorbar(f)
        
        ax.set_title('Manifold latent for latent: ' + latent_name)
        if figname is not None:
            figname1 = figname + 'repr_manifold_pca' 
            figname1 += '_varied='+ misc.ints_to_str(vary_latents)
            figname1 += '_true='+ misc.ints_to_str(latent) + '.pdf'
            plt.savefig(figname1)
            logger.info(f'Figure saved {figname1}')
        if config.log_wandb:
            wandb.log({f'plot/manifold_repr'+
                       f'_varied{misc.ints_to_str(vary_latents)}_col{i}':\
                                                        wandb.Image(plt)})
        plt.close(fig)


T_SERIES = ["bce_loss", "kl_loss", "learned_alpha"]
def plot_curves(shared,config,logger,figname=None,val_name=None):
    if config.plot_on_black:
        plt.style.use('dark_background')
    epochs = np.arange(len(vars(shared)[T_SERIES[0]]))*config.val_epoch
    for key in T_SERIES:
        if key in shared:
            fig, ax = plt.subplots(figsize=(8,7))
            ax.plot(epochs,vars(shared)[key])
            ax.set_xlabel('epochs')
            ax.set_ylabel(key)
            if figname is not None:
                figname1 = figname + 'curve_' + key + '.pdf'
                plt.savefig(figname1)
            plt.close()


def plot_matrix(example_R,a, config, logger, figname=None):
    if config.plot_on_black:
        plt.style.use('dark_background')
    
    
    for i in range(a.shape[0]):
        action = a[i]
        R = np.around(example_R[i],decimals=2)
            
        fig, ax = plt.subplots(figsize=(4,4))
        # kwargs={'vmin':-2,'vmax':2,'cmap':'gray'}
        M = np.abs(R).max()
        # norm = mcolors.TwoSlopeNorm(vcenter=0,vmin=-M,vmax=M)
        # im = ax.imshow(R,cmap='bwr',norm=norm)
        im = ax.imshow(R,cmap='bwr',vmin=-M,vmax=M)

        
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('value', rotation=-90, va="bottom")

        ax.axis('off')

        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                im.axes.text(j, i, R[i,j])

        if figname is not None:
            figname1= figname + f'action:{action}.pdf'
            plt.savefig(figname1,bbox_inches='tight')
            logger.info(f'Figure saved {figname1}')
        if config.log_wandb:
            log_dict= {
                f'val/learned_representation/action:{action}':wandb.Image(plt)}
            wandb.log(log_dict)
        plt.close(fig)
    
def plot_step_recon_loss(step_losses, config, figname=None):
    if config.plot_on_black:
        plt.style.use('dark_background')

    fig, ax = plt.subplots(figsize=(8, 7))
    mean, se = np.mean(step_losses, axis=0), sem(step_losses, axis=0)
    ax.plot(np.arange(mean.shape[0]), mean, color='C0')
    ax.fill_between(np.arange(mean.shape[0]), mean - se, mean + se, color='C0', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reconstruction Loss')
    if figname is not None:
        figname1 = figname + 'step_recon.pdf'
        plt.savefig(figname1)
    plt.close()

def plot_thetas(dhandler, nets: aeprod.AutoencoderProdrep, config, logger,
                figname=None):
    if config.plot_on_black:
        plt.style.use('dark_background')

    reps = nets.grp_morphism.action_reps
    dim = nets.grp_morphism.dim_representation
    x = np.arange(len(reps[0].thetas))
    xticks = []
    for i in range(1, dim + 1):
        for j in range(i + 1, dim + 1):
            xticks.append(f'{i}{j}')
    width = 0.5

    nrows = dhandler.action_dim + 1
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * 2, nrows * 2))
    # kwargs={'vmin':0,'vmax':1,'cmap':'gray'}
    for n in range(dhandler.action_dim):
        for sign in range(2):

            a = torch.zeros(dhandler.action_dim, dtype=int)
            a[n] = 1 - 2 * sign
            id = dhandler.transition_to_index(a)

            thetas = reps[id].thetas.to('cpu').data.numpy()
            # kwargs = {}
            axes[n, sign].bar(x - width / 2, thetas / (2 * np.pi), label='Rep {}'.format(a))
            axes[n, sign].set_ylim(-.5, .5)
            axes[n, sign].axhline(0)
            axes[n, sign].set_xticks(x - 0.25)
            axes[n, sign].set_xticklabels(xticks)
            axes[n, sign].set_xlabel('$ij$')
            axes[n, sign].set_ylabel(r"$\theta / 2\pi$")
            axes[n, sign].set_title(f"${a.numpy()}$")

    a = torch.zeros(dhandler.action_dim, dtype=int)
    id = udutils.action_to_id(a)
    thetas = reps[id].thetas.to('cpu').data.numpy()
    axes[-1, 0].bar(x - width/2, thetas/(2*np.pi), label='Rep {}'.format(a))
    axes[-1, 0].set_ylim(-.5, .5)
    axes[-1, 0].axhline(0)
    axes[-1, 0].set_xticks(x-0.25)
    axes[-1, 0].set_xticklabels(xticks)
    axes[-1, 0].set_xlabel('$ij$')
    axes[-1, 0].set_ylabel(r"$\theta / 2\pi$")
    axes[-1, 0].set_title(f"${a.numpy()}$")

    axes[-1, -1].axis('off')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    if figname is not None:
        figname1 = figname + 'thetas.pdf'
        plt.savefig(figname1)
        logger.info(f'Figure saved {figname1}')
    if config.log_wandb:
        wandb.log({'plot/grp_repr':wandb.Image(plt)})
    plt.close(fig)

    # TODO
    pass
    # for rep in nets.grp_morphism.actions_reps:

    #     fig, ax = plt.subplots(figsize=(8,7))

    #         ax.plot(epochs,vars(shared)[key])
    #         ax.set_xlabel('epochs')
    #         ax.set_ylabel(key)
    #         if figname is not None:
    #             figname1 = figname + 'curve_' + key + '.pdf'
    #             plt.savefig(figname1)
    #         plt.close()
