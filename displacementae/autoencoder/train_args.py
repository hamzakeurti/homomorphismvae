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
# @title          :displacementae/autoencoder/train_args.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/11/2021
# @version        :1.0
# @python_version :3.7.4

import argparse
from datetime import datetime

def parse_cmd_arguments(repr):
    curr_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    description = 'Geometric Autoencoder'
    dout_dir = './out/run_'+curr_date
    parser = argparse.ArgumentParser(description=description)
    data_args(parser)
    train_args(parser)
    net_args(parser)
    misc_args(parser,dout_dir)
    if repr == 'prodrepr':
        prodrepr_args(parser)
    config = parser.parse_args()
    return config

def data_args(parser):
    dgroup = parser.add_argument_group('Data options')
    dgroup.add_argument('--dataset', type=str, default='armeye', 
                        help='Name of dataset',choices=['armeye','dsprites'])
    dgroup.add_argument('--n_joints', type=int, default=3,
                        help='Number of joints in the robot')
    dgroup.add_argument('--fixed_in_sampling', type=str, default='',
                        help='indices of fixed joints in sampling')
    dgroup.add_argument('--fixed_values', type=str, default='', 
                        help='Values of fixed joints')
    dgroup.add_argument('--fixed_in_intervention', type=str, default='', 
                        help='Indices of fixed joints in intervention')
    dgroup.add_argument('--intervene', action='store_true',
                        help='Whether to vary joint positions.')
    dgroup.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle the dataset.')
    dgroup.add_argument('--displacement_range', type=str,
                        default="-3,3", help='Range of uniform distribution '+
                        'from which to sample future joint position')
    dgroup.add_argument('--distrib', type=str, default='uniform', 
                        help='Displacement sampling distribution',
                        choices=['uniform','disentangled'])
    dgroup.add_argument('--data_root', type=str, 
                        help='Root directory of the dataset directory.')
    dgroup.add_argument('--data_random_seed', default=42,type=int,
                        help='Specify data random seed for reproducibility.')
    dgroup.add_argument('--num_train', type=int, default=100,
                        help='Number of training samples')
    dgroup.add_argument('--num_val', type=int, default=15,
                        help='Number of evaluation samples')
    dgroup.add_argument('--cyclic_trans', action='store_true',
                        help='considers position as a cyclic latent.')
    

def train_args(parser):
    """
    Arguments specified in this function:
            - `batch_size`
            - `n_iter`
            - `epochs`
            - `lr`
            - `momentum`
            - `weight_decay`
            - `use_adam`
    """
    tgroup = parser.add_argument_group('Train options')
    tgroup.add_argument('--batch_size',type=int,default=50, 
                        help='Training batch size')
    tgroup.add_argument('--n_iter', type=int, default=10000,
                        help='Number of training iterations per task. ')
    tgroup.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs.')
    tgroup.add_argument('--val_epoch', type=int, default=5,
                        help='Number of epochs between evaluations.')
    tgroup.add_argument('--plot_epoch', type=int, default=200,
                        help='Number of epochs between evaluations.')
    tgroup.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate of optimizer(s).')
    tgroup.add_argument('--momentum', type=float, default=0,
                        help='Momentum of optimizer(s) (Only used for SGD).')
    tgroup.add_argument('--weight_decay', type=float, default=0,
                        help='Weight decay of optimizer(s).')
    tgroup.add_argument('--use_adam', action='store_true',
                        help='Use Adam optimizer instead of SGD.')

def net_args(parser):
    """
    Arguments specified in this function:
            - `net_act`
            - `conv_channels`
            - `lin_channels`
            - `kernel_sizes`
            - `strides`
            - `learn_geometry`
            - `specified_grp_step`
            - `variational`
            - `beta`
            - `n_free_latents`
    """
    ngroup = parser.add_argument_group('network options')
    ngroup.add_argument('--net_act',type=str,default='relu',
                        choices=['sigmoid','relu','tanh','none'], 
                        help='Training batch size')
    ngroup.add_argument('--conv_channels', type=str, default='',
                        help='Channels per layer. '+
                        'Input channels must be included')
    ngroup.add_argument('--lin_channels', type=str, default='',
                        help='linear channels.')
    ngroup.add_argument('--kernel_sizes', type=str, default='5',
                        help='kernel sizes of convolution layers.')
    ngroup.add_argument('--strides', type=str, default='1',
                        help='strides of convlution layers.')
    ngroup.add_argument('--learn_geometry',action='store_true', 
                        help='Whether to learn the grp action parameters. '+
                        'If not, these should be provided in arg '+
                        '--specified_grp_step')
    ngroup.add_argument('--specified_grp_step', type=str, default='0', 
                        help='specified grp action parameters')
    ngroup.add_argument('--variational',action='store_true', 
                        help='Whether the network outputs ' + 
                        'should be considered as mean and var of a gaussian.')
    ngroup.add_argument('--beta',type=float, 
                        help='Beta factor of the beta-VAE ' +
                        'balances contribution of prior matching loss. ' + 
                        'Defaults to 1.')
    ngroup.add_argument('--n_free_units',type=int, default=0, 
                        help='number of representation units ' +
                        'not transformed by the action representations.')
    ngroup.add_argument('--spherical',action='store_true', 
                        help='If True, the representation vector is ' + 
                        'normalized prior to being forwarded to transform or ' +
                        'decoder')

def misc_args(parser,dout_dir=None):
    if dout_dir is None:
        dout_dir = './out/run_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    mgroup = parser.add_argument_group('Miscellaneous options')
    mgroup.add_argument('--out_dir', type=str, default=dout_dir,
                        help='Where to store the outputs of this simulation.')
    mgroup.add_argument('--use_cuda', action='store_true',
                        help='Whether to use GPU.')
    mgroup.add_argument('--cuda_number', type=int,
                        help='if use_cuda, GPU device number.')
    mgroup.add_argument('--random_seed', type=int, default=42,
                        help='Specify random seed for reproducibility')
    mgroup.add_argument('--plot_on_black', action='store_true', 
                        help='Whether to plot using dark background style')
    mgroup.add_argument('--no_plots', action='store_true', 
                        help='Whether to plot figures')
    mgroup.add_argument('--plot_manifold_latent', type=str, default='',
                        help='Which latent units to visualize the manifold of.')
    mgroup.add_argument('--plot_vary_latents', type=str, default='',
                        help='Which joints to move' + 
                        'to produce a manifold of latents.')
    mgroup.add_argument('--checkpoint', action='store_true', 
                        help='Saves a checkpoint of the model and optimizer '+
                        'at the end of training')
    mgroup.add_argument('--plot_pca', action='store_true', 
                        help='Plots scatter of representations projected '+
                             'along 2 main pca components')
    
def prodrepr_args(parser):
    pgroup = parser.add_argument_group('Prod Repr options')
    pgroup.add_argument('--repr_dim', type=int, 
                        help='Dimensionality of the group representation.')
    pgroup.add_argument('--entanglement_loss', action='store_true', 
                        help='Whether to use the entanglement loss.')
    pgroup.add_argument('--entanglement_weight', type=float, default=0.01, 
                        help='Weight of the entanglement loss.')
    pgroup.add_argument('--plot_thetas', action='store_true', 
                        help='Whether to plot the learned thetas.')
