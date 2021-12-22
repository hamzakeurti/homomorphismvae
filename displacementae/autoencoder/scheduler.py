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



class Scheduler():
    def __init__(self, toggle_every = 10):
        self.toggle_every = toggle_every

    def toggle_train(self, nets1, nets2, epoch):
        """
        Switches 
        """ 
        if (epoch//2) % self.toggle_every:
            for net1 in nets1:
                toggle_grad(net1,True)
            for net2 in nets2:
                toggle_grad(net2,False)
        else:
            for net1 in nets1:
                toggle_grad(net1,False)
            for net2 in nets2:
                toggle_grad(net2,True)

def toggle_grad(model, on = True):
    for p in model.parameters():
        p.requires_grad = on
