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

from typing import List, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import torch
from argparse import Namespace
import wandb
import os


import displacementae.networks.autoencoder_prodrep as aeprod
import displacementae.utils.misc as misc
import displacementae.utils.data_utils as udutils
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


def plot_n_step_reconstruction(imgs, actions, nets, device, logger, 
                               plot_on_black:bool=False, n_steps:int=1, 
                               n_examples:int=7, savefig:bool=False, 
                               savedir:str='', log_wandb:bool=False, 
                               figname:str="reconstructions", 
                               epoch:Optional[int]=None):
    # always reonstruct first, even when not considered in loss
    if plot_on_black:
        plt.style.use('dark_background')

    X1 = torch.FloatTensor(imgs[:,0]).to(device)

    dj = torch.FloatTensor(actions).to(device)

    h, _, _, mu, logvar = nets(X1, dj)
    Xi_hat = torch.sigmoid(h)

    nrows = n_examples
    ncols = 2 + 2*n_steps

    unit_length = 1.5

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * unit_length,
                                      nrows * unit_length))
    kwargs = {'vmin': 0, 'vmax': 1}
    
    Xi = imgs
    if imgs.shape[2] == 1:
        kwargs['cmap'] = 'gray'
        Xi = Xi[:,:,0]
        Xi_hat = Xi_hat[:,:,0].cpu().numpy()
        X1 = X1[:,0].cpu().numpy()
    else:
        Xi = np.moveaxis(Xi,-3,-1)
        Xi_hat = np.moveaxis(Xi_hat.cpu().numpy(),-3,-1)
        X1 = np.moveaxis(X1.cpu().numpy(),-3,-1)

    for row in range(nrows):
        for i in range(Xi_hat.shape[1]):
            axes[row,2*i].imshow(Xi[row,i],**kwargs)#should be i+1
            axes[row,2*i+1].imshow(Xi_hat[row,i],**kwargs)
        if plot_on_black:
            for j in range(ncols):
                axes[row, j].axes.xaxis.set_visible(False)
                axes[row, j].axes.yaxis.set_visible(False)
        else:
            for j in range(ncols):
                axes[row, j].axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.1)

    title = figname.split('.')[0]
    if savefig:
        if epoch is not None:
            figname = f'{epoch} - {figname}'
        savepath = os.path.join(savedir,figname)
        plt.savefig(savepath)
        logger.info(f'Figure saved {savepath}')
    if log_wandb:
        wandb.log({f'plot/reconstructions/{title}':wandb.Image(plt)})
    plt.close(fig)



def plot_supervised_reconstruction(dhandler, nets, config, device, logger, figname):

    if config.plot_on_black:
        plt.style.use('dark_background')


    nrows = 5
    ncols = 2

    x, z = [torch.FloatTensor(elem).to(device) for elem in dhandler.get_val_batch()]
    x, z = x[:nrows], z[:nrows]

    x_hat = torch.sigmoid(nets(z))

    nrows = 5
    ncols = 2

    unit_length = 1.5

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * unit_length,
                                      nrows * unit_length))
    kwargs = {'vmin': 0, 'vmax': 1}
    if x.shape[1] == 1:
        kwargs['cmap'] = 'gray'
        x = x[:,0].cpu().numpy()
        x_hat = x_hat[:,0].cpu().numpy()
    else:
        x = np.moveaxis(x.cpu().numpy(),-3,-1)
        x_hat = np.moveaxis(x_hat.cpu().numpy(),-3,-1)

    for row in range(nrows):  
        axes[row,0].imshow(x[row],**kwargs)
        axes[row,1].imshow(x_hat[row],**kwargs)
        
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


def plot_rollout_reconstructions(imgs, actions, nets, device, logger, 
                                 n_rollouts:int=7, powers:bool=False, 
                                 n_images:Optional[int]=None, 
                                 savefig:bool=False, savedir:str='', 
                                 log_wandb:bool=False, 
                                 figname:str="rollouts_reconstructions",
                                 epoch:Optional[int]=None):
    """
    Plots the reconstructions of the first `n_{rollouts}` rollouts.

    """

    X = torch.FloatTensor(imgs).to(device)
    a = torch.FloatTensor(actions).to(device)

    if powers:
        indices = np.power(2, np.arange(2,np.log2(a.shape[1]),1)).astype(int)
    else:
        indices = np.arange(0,a.shape[1],1).astype(int)
    
    if n_images is not None:
        indices = indices[:n_images]
    
    n_steps = len(indices)

    n_rows = X.shape[0]*2
    n_cols = n_steps

    # Forward pass
    X_hat, _,_,_,_ = nets(X[:,0], a) 
    X_hat = torch.sigmoid(X_hat)

    unit_length = 2
    fig, axs = plt.subplots(n_rows, n_cols, 
                            figsize=(n_cols*unit_length, n_rows*unit_length))
    
    kwargs = {'vmin': 0, 'vmax': 1}
    if X.shape[2] == 1:
        kwargs['cmap'] = 'gray'
        X = X[:,:,0].cpu().numpy()
        X_hat = X_hat[:,:,0].cpu().numpy()
    else:
        X = np.moveaxis(X.cpu().numpy(),-3,-1)
        X_hat = np.moveaxis(X_hat.cpu().numpy(),-3,-1)
    
    for row in range(n_rollouts):
        for col in range(n_cols):
            axs[row*2,col].imshow(X[row,indices[col]], **kwargs)
            axs[row*2,col].axis('off')
            axs[row*2+1,col].imshow(X_hat[row,indices[col]], **kwargs)
            axs[row*2+1,col].axis('off')
    
    plt.subplots_adjust(wspace=0, hspace=0.1)

    title = figname.split('.')[0]
    if savefig:
        if epoch is not None:
            figname = f'{epoch} - {figname}'
        savepath = os.path.join(savedir,figname)
        plt.savefig(savepath)
        logger.info(f'Figure saved {savepath}')
    if log_wandb:
        wandb.log({f'plot/rollouts/{title}':wandb.Image(plt)})
    plt.close(fig)


def plot_manifold(representations, true_latents, logger, plot_on_black:bool=False, 
                  log_wandb:bool=False, label:str='', savedir:str='', 
                  savefig:bool=False, figname:str="manifold", 
                  epoch:Optional[int]=None):
    kwargs={}
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if plot_on_black:
        plt.style.use('dark_background')
    if plot_on_black:
        kwargs={'cmap':'summer'}

    kwargs['alpha']=0.8 # type: ignore
    kwargs['edgecolors']='none'


    fig, ax = plt.subplots(figsize=(8,7))
    # check if true_latents is squeezable to 1D array
    representations = np.squeeze(representations)
    
    if len(representations.shape) == 1:
        f = ax.scatter(x=true_latents, y=representations)
        ax.set_xlabel(f'{label}', fontsize=ts)
        ax.set_ylabel(f'repr unit', fontsize=ts)
    elif representations.shape[1] == 2:
        f = ax.scatter(x=representations[:,0], y=representations[:,1], 
                       c=true_latents, **kwargs)
        ax.set_xlabel(f'repr unit 0', fontsize=ts)
        ax.set_ylabel(f'repr unit 1', fontsize=ts)
        plt.colorbar(f)

    title = figname.split('.')[0]
    ax.set_title(title)
    if savefig:
        if epoch is not None:
            figname = f'{epoch} - {figname}'
        savepath = os.path.join(savedir,figname)
        plt.savefig(savepath)
        logger.info(f'Figure saved {savepath}')
    if log_wandb:
        wandb.log({f'plot/manifold/{title}':wandb.Image(plt)})
    plt.close(fig)


def plot_manifold_markers(latents_clr, latents_mrk, representations, logger,
                          plot_on_black:bool=False, log_wandb:bool=False, 
                          savedir:str=None, savefig:bool=False, 
                          figname:str="manifold",
                          epoch:Optional[int]=None):
    """
    Plots a scatter plot of the representation manifold with 
    colors and markers corresponding to the true latents.

    Args:
        latents_clr (np.ndarray): array of true latents to be used as colors.
        latents_mrk (np.ndarray): array of true latents to be used as markers.
        representations (np.ndarray): array of representations to be plotted.
        logger (logging.Logger): logger object.
        plot_on_black (bool, optional): whether to plot on black background.
            Defaults to False.
        log_wandb (bool, optional): whether to log to wandb. Defaults to False.
        figname (str, optional): figure name. Defaults to None.
        savefig (bool, optional): whether to save figure. Defaults to False.
    """
    
    ts, lw, ms = _DEFAULT_PLOT_CONFIG
    if plot_on_black:
        plt.style.use('dark_background')
        kwargs={'cmap':'summer'}
    else:
        kwargs={}

    kwargs['alpha'] = 0.5
    kwargs['edgecolors']='none'
    
    kwargs['vmin'] = min(latents_clr)
    kwargs['vmax'] = max(latents_clr)
        
    fig, ax = plt.subplots(figsize=(8,7))

    # f = ax.scatter(x=latent2d[:,0], y=latent2d[:,1], c=latents[:,i],
                    # **kwargs)
    for x,y,c,m in zip(
            representations[:,0],representations[:,1],latents_clr,latents_mrk):
        f = ax.scatter(x=x, y=y, c=c,marker=MARKERS[m%len(MARKERS)],
                    **kwargs)

    ax.set_xlabel(f'latent component 0', fontsize=ts)
    ax.set_ylabel(f'latent component 1', fontsize=ts)
    
    plt.colorbar(f)
    
    title = figname.split('.')[0]
    ax.set_title(title)
    if savefig:
        if epoch is not None:
            figname = f'{epoch} - {figname}'
        savepath = os.path.join(savedir,figname)
        plt.savefig(savepath)
        logger.info(f'Figure saved {savepath}')
    if log_wandb:
        wandb.log({f'plot/manifold/{title}':wandb.Image(plt)})
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
        # kwargs={'vmin':-1,'vmax':1,'cmap':'gray'}
        # M = np.abs(R).max()
        M = 1
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
                f'val/learned_representation/action={action}':wandb.Image(plt)}
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
    
    if not isinstance(nets,aeprod.AutoencoderProdrep):
        raise NotImplementedError("--plot_thetas is specific to the prodrepr ")
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