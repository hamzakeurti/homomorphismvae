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

import displacementae.utils.args as args

from displacementae.grouprepr.representation_utils import Representation

def parse_cmd_arguments(representation=Representation.BLOCK_ROTS ,description=''):
    curr_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if not description:
        description = 'Geometric Autoencoder'
    dout_dir = './out/run_'+curr_date
    parser = argparse.ArgumentParser(description=description)
    args.data_args(parser)
    args.train_args(parser)
    args.net_args(parser)
    args.misc_args(parser,dout_dir)
    args.group_repr_args(parser, representation)

    config = parser.parse_args()
    return config

