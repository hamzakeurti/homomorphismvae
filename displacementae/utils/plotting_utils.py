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
import numpy as np
import torch
from argparse import Namespace

import utils.misc as misc


_DEFAULT_PLOT_CONFIG = [12, 5, 8] # fontsize, linewidth, markersize

# Simple 2D regression grid.
_TWO_D_MISC = Namespace()
_TWO_D_MISC.x_range = [-5, 5]
_TWO_D_MISC.y_range = [-5, 5]

def plot(dhandler,nets,shared,config,logger):
    
    pass

def plot_reconstruction(dhandler,nets,shared,config,logger):
    batch = dhandler.get_val_batch()

    pass

def plot_manifold(dhandler, nets, shared, config, device, logger, mode,
                epoch, vary_joints=[3], plot_latent=[0,1], figname=None):
    """
    Produces colored scatter plot of the latent representation of 
    the different positions in the joint space.

    A 1D or 2D grid of joint positions are generated, 
    corresponding images are forwarded through the encoder.
    Resulting latent units are 
    """
    ts, lw, ms = _DEFAULT_PLOT_CONFIG

    encoder = nets[0]

    indices = dhandler.get_indices_vary_joints(vary_joints)
    labels = dhandler._classes[indices][:,vary_joints].squeeze()
    batch_size = config.batch_size
    n_batches = len(indices) // batch_size + 1
    
    results = []

    for i in range(n_batches):
        batch_indices = indices[ i * batch_size : (i+1) * batch_size]
        images, _ = dhandler.get_images_batch(batch_indices)
        X = torch.FloatTensor(images).to(device)
        with torch.no_grad():
            h = encoder(X)
            results.append(h[:,plot_latent].cpu().numpy())
    results = np.vstack(results).squeeze()

    fig, ax = plt.subplots(figsize=(8,7))

    if len(plot_latent) == 1:
        f = ax.scatter(x=labels, y=results)
        ax.set_xlabel('true label', fontsize=ts)
        ax.set_ylabel('latent', fontsize=ts)
    if len(plot_latent) == 2:
        f = ax.scatter(x=results[:,0], y=results[:,1], c=labels)
        ax.set_xlabel('latent 0', fontsize=ts)
        ax.set_ylabel('latent 1', fontsize=ts)
        ax.set_xlim(_TWO_D_MISC.x_range)
        ax.set_ylim(_TWO_D_MISC.y_range)
        plt.colorbar(f)
    
    figname += 'repr_manifold_latent=' + misc.ints_to_str(plot_latent) 
    figname += '_true='+ misc.ints_to_str(vary_joints) + '.pdf'
    plt.savefig(figname)
    logger.info(f'Figure saved {figname}')
    pass

def plot_curve(dhandler,nets,shared,config,logger,val_name):
    pass