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
from torch.utils.data import Dataset

class TransitionDataset(Dataset):
    def __init__(self, rseed=None, transitions_on=True, n_transitions=None):
        
        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random
        self._rand = rand
        self._rseed = rseed

        # Number of samples
        self.num_train = 0
        self.num_val = 0

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

        # Number of transitions
        if not transitions_on:
            self.n_transitions = 0
        elif n_transitions is None:
            self.n_transitions = 1
        else:
            self.n_transitions = n_transitions
        

    def __getitem__(self,idx):
        pass

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
        indices = np.empty(shape=(idx.shape[-1],self.n_transitions+1), dtype=int)
        transitions = np.empty(shape=(idx.shape[-1],self.n_transitions,self.n_latents))
        indices[:,0] = idx
        for i in range(self.n_transitions):
            idx2, dj = self.transition(indices[:,i])
            indices[:,i+1]= idx2
            transitions[:,i] = dj
        return indices,transitions
    
    def get_val_batch(self):
        pass