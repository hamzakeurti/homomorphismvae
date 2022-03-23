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
# @title          :displacementae/utils/train_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :09/02/2022
# @version        :1.0
# @python_version :3.7.4

from argparse import Namespace

import utils.sim_utils as sim_utils
import data.data_utils as data_utils
import networks.network_utils as net_utils 

def run(mode='autoencoder',repr='blockrepr'):
    """Script for setting up and launching the training of the models.

    Args:
        mode (str): architecture type, supports 'autoencoder', defaults to 
            'autoencoder'
        repr (str): group representation, defaults to 'blockrepr'.
            'blockrepr': actions are represented by block diagonal matrices of 
            2D rotation matrices. 
    """
    # Mode dependent imports
    if mode == 'autoencoder':
        import autoencoder.train_args as train_args
        import autoencoder.train_utils as tutils
    elif mode == 'homomorphism':
        import homomorphism.train_args as train_args
        import homomorphism.train_utils as tutils

    # parse commands
    config = train_args.parse_cmd_arguments()
    # setup environment
    device, logger = sim_utils.setup_environment(config)
    sim_utils.backup_cli_command(config)
    # setup dataset
    dhandler, dloader = data_utils.setup_data(config, mode)
    # setup models
    nets = net_utils.setup_network(config, dhandler, device, mode=mode,
        repr=repr)
    # setup shared
    shared = Namespace()

    logger.info('### Training ###')
    finished_training = tutils.train(dhandler, dloader, nets,
                              config, shared, device, logger, mode)
    return