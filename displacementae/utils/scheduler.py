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
# @title          :displacementae/autoencoder/scheduler.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :21/12/2021
# @version        :1.0
# @python_version :3.7.4

from argparse import Namespace
import utils.misc as misc
import torch.nn as nn
from typing import List


class Scheduler():
    def __init__(self,  grp1: List[nn.Module], grp2: List[nn.Module], toggle_every=[10, 10]):
        self.toggle_every = toggle_every
        self.counter = 0
        self.grp1 = grp1
        self.grp2 = grp2
        for net2 in self.grp2:
            toggle_grad(net2, False)
        for net1 in self.grp1:
            toggle_grad(net1, True)

    def toggle_train(self):
        """
        Switches requires grad on/off every `toggle_every` epochs.
        """
        if self.counter == self.toggle_every[0]:
            for net1 in self.grp1:
                toggle_grad(net1, False)
            for net2 in self.grp2:
                toggle_grad(net2, True)
        elif self.counter == sum(self.toggle_every):
            for net2 in self.grp2:
                toggle_grad(net2, False)
            for net1 in self.grp1:
                toggle_grad(net1, True)
            self.counter = 0
        # if (epoch//2) % self.toggle_every == 0:
        #     for net2 in nets2:
        #         toggle_grad(net2,False)
        #     for net1 in nets1:
        #         toggle_grad(net1,True)
        # else:
        #     for net1 in nets1:
        #         toggle_grad(net1,False)
        #     for net2 in nets2:
        #         toggle_grad(net2,True)
        self.counter += 1


def toggle_grad(model, on=True):
    for p in model.parameters():
        p.requires_grad = on


def setup_scheduler(config: Namespace, group1: list, group2: list) -> Scheduler:
    toggle_every = misc.str_to_ints(config.toggle_training_every)
    if isinstance(toggle_every, int):
        toggle_every = [toggle_every, toggle_every]

    return Scheduler(group1, group2, toggle_every)
