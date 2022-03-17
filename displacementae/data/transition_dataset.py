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
# @title          :displacementae/data/dsprites.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/11/2021
# @version        :1.0
# @python_version :3.7.4
"""
Abstract Dataset class for transitions tuple :math:`(o_t,g_t,o_{t+1})`.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

import numpy as np

class TransitionDataset:
    def __init__(self, root, num_train, num_val, rseed=None):
        
        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random
        self._rand = rand
        self._rseed = rseed

        # Number of samples
        self.num_train = num_train
        self.num_val = num_val

        # Latents Config
        self.transitions_on = True
        self.n_latents = 0
        self.latents = np.array([])

        self.fixed_in_sampling = []
        self.fixed_values = []
        self.varied_in_sampling = []

        self.fixed_in_action = []
        self.varied_in_action = []
        self.transition_range = []

        
        # data root
        self.root = root

    def __getitem__(self,idx):
        images, latents, dj = [], [], []

        indices = self.train_idx[idx]
        for index in indices:
            images.append(self.images[index])
            latents.append(self.latents[index])  
        dj = self.train_dj[idx]
        return images, latents, dj

    def __len__(self):
        pass

    def latents_2_index(self,latents):
        """
        Converts a vector of latents values to its index in the subdataset.
        """
        return np.dot(
            latents[...,self.varied_in_sampling],self.latent_bases_varied)

    def setup_latents_bases(self):
        """
        Computes the latents bases vector for converting latents vectors to indices.
        """
        self.num_latents_varied = self.num_latents[self.varied_in_sampling]
        self.latent_bases = np.concatenate([
            np.cumprod(self.num_latents[::-1])[::-1][1:],[1]])
        self.latent_bases_varied = np.concatenate([
            np.cumprod(self.num_latents_varied[::-1])[::-1][1:],[1]])    
        self.dataset_size = np.prod(self.num_latents_varied)

    def transition(self, idx):
        pass

    def observe_n_transitions(self,idx):
        # TODO
        pass
