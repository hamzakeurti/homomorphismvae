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
# @title          :displacementae/data/data_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :12/11/2021
# @version        :1.0
# @python_version :3.7.4

from torch.utils.data import DataLoader

from data.dsprites import DspritesDataset
from data.trajs import TrajectoryDataset
import utils.misc as misc


def setup_data(config, mode='autoencoder'):
    if config.train_trajs is not None:
        return setup_trajectory_dataset(config)
    elif config.dataset == 'dsprites':
        return setup_dsprites_dataset(config, mode)


def setup_dsprites_dataset(config, mode='autoencoder'):
    fixed_in_sampling = misc.str_to_ints(config.fixed_in_sampling)
    fixed_values = misc.str_to_ints(config.fixed_values)
    fixed_in_action = misc.str_to_ints(config.fixed_in_intervention)
    action_range = misc.str_to_ints(config.displacement_range)
    if mode == 'homomorphism':
        config.intervene = True
    if mode == 'autoencoder':
        config.n_steps = 1
    dhandler = DspritesDataset(
        root=config.data_root,
        num_train=config.num_train,
        num_val=config.num_val,
        rseed=config.data_random_seed,
        fixed_in_sampling=fixed_in_sampling,
        fixed_values=fixed_values,
        fixed_in_action=fixed_in_action,
        transitions_on=config.intervene,
        n_transitions=config.n_steps,
        action_range=action_range,
        cyclic_trans=config.cyclic_trans,
        dist=config.distrib,
        return_integer_actions=config.integer_actions,
        rotate_actions=config.rotate_actions,
        )
    dloader = DataLoader(
        dataset=dhandler, batch_size=config.batch_size,
        shuffle=config.shuffle)
    return dhandler, dloader


def setup_trajectory_dataset(config):

    dhandlers = [
        TrajectoryDataset(config.train_trajs, config.dataset, config.data_random_seed),
        TrajectoryDataset(config.valid_trajs, config.dataset, config.data_random_seed)
    ]
    dloaders = [
        DataLoader(dataset=dhandlers[0], batch_size=config.batch_size, shuffle=config.shuffle),
        DataLoader(dataset=dhandlers[1], batch_size=10, shuffle=False),
    ]

    return dhandlers, dloaders
