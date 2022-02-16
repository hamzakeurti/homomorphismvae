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

from data.dsprites import DspritesDataset, FixedJointsSampler
import utils.misc as misc

def setup_data(config):
    if config.dataset == 'dsprites':
        return setup_dsprites_dataset(config)

def setup_dsprites_dataset(config):
    fixed_in_sampling = misc.str_to_ints(config.fixed_in_sampling)
    fixed_values = misc.str_to_ints(config.fixed_values)
    fixed_in_intervention = misc.str_to_ints(config.fixed_in_intervention)
    intervention_range = misc.str_to_ints(config.displacement_range)
    dhandler = DspritesDataset(
        root=config.data_root,
        num_train=config.num_train,
        num_val=config.num_val, 
        rseed=config.data_random_seed, 
        fixed_in_sampling=fixed_in_sampling, 
        fixed_values=fixed_values, 
        fixed_in_intervention=fixed_in_intervention, 
        intervene=config.intervene, intervention_range=intervention_range,
        cyclic_trans=config.cyclic_trans, dist=config.distrib)
    dloader = DataLoader(
        dataset=dhandler, batch_size=config.batch_size,
        shuffle=config.shuffle)
    return dhandler, dloader