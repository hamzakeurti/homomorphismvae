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
Abstract Dataset class for transitions tuple :math:`(o_1,g_1,...,g_{n-1},o_n)`.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from typing import Tuple, List

import numpy as np
from torch.utils.data import Dataset


class TransitionDataset(Dataset):
    """Abstract dataset for transitions.

    :param rseed: Random seed for sampling operations of this class,
                defaults to None
    :type rseed: int, optional
    :param n_transitions: Number of transitions in an interaction sequence. 
                        For instance, if it is 1 then samples are tuples where 
                        the first element is a numpy array of 
                        2 observations o_1 and o_2, and the second element is a 
                        numpy array with a transition signal (/action) g. 
                        Defaults to 1
    :type n_transitions: int, optional
    """


    def __init__(self, rseed:int=None, n_transitions:int=1):
        
        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random
        self._rand = rand
        self._rseed = rseed

        # Number of transitions
        self.n_transitions = n_transitions
    

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray]:
        """Accesses the :math:`i^{th}` sample (transition sequence) in the dataset.

        :param idx: index
        :type idx: int
        :return: a numpy array of observations and 
                 a numpy array of transition signals.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        pass

    def __len__(self):
        pass

    def resample_data(self) -> None:
        """resamples the training dataset. (Does nothing for some datasets).
        """
        pass

    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of evaluation samples.

        :return: a tuple of a batch of observation evaluation samples, 
                 a batch of their associated latent representations and 
                 a batch of transition evaluation samples. 
        :rtype: Tuple[np.ndarray, np.ndarray, np.ndarray]
        """
        pass

    def get_example_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """returns a set of example actions (transition signals) with labels. 

        :return: a tuple of a batch of action signals as perceived by the agent 
                 and associated labels.
        :rtype: Tuple[np.ndarray, np.ndarray]

        .. todo::
        maybe get rid of the labels.
        """
        pass
    
    @property
    def action_units(self) -> int:
        """Number of action units 

        :return: Dimension of the action vector.
        :rtype: int
        """
        pass

    @property
    def in_shape(self) -> List[int]:
        """The shape of the observations :math:`o_t`.

        :return: A list of the dimensions of an observation sample.
                 Also contains number of channels.
                 For instance for levels of gray images, 
                 this returns `[1, height, width]`. 
        :rtype: List[int]
        """
        pass


    @property
    def num_train(self) -> int:
        """Number of training samples

        :return: `int` indicating total number fo training samples.
        :rtype: int
        """
        pass

    @property
    def num_val(self) -> int:
        """Number of evaluation samples

        :return: an integer indicating the number of evaluation samples.
        :rtype: int
        """
        pass

# class TransitionDataset(Dataset):
#     def __init__(self, rseed=None, transitions_on=True, n_transitions=None):
        
#         # Random generator
#         if rseed is not None:
#             rand = np.random.RandomState(rseed)
#         else:
#             rand = np.random
#         self._rand = rand
#         self._rseed = rseed

#         # Number of samples
#         self.num_train = 0
#         self.num_val = 0

#         # Latents Config
#         self.transitions_on = transitions_on
#         self.n_latents = 0
#         self.latents = np.array([])

#         self.fixed_in_sampling = []
#         self.fixed_values = []
#         self.varied_in_sampling = []

#         self.fixed_in_action = []
#         self.varied_in_action = []
#         self.transition_range = []

#         # Number of transitions
#         if not transitions_on:
#             self.n_transitions = 0
#         elif n_transitions is None:
#             self.n_transitions = 1
#         else:
#             self.n_transitions = n_transitions
        

#     def __getitem__(self,idx):
#         pass

#     def __len__(self):
#         pass

#     def latents_2_index(self,latents):
#         """
#         Converts a vector of latents values to its index in the subdataset.
#         """
#         return np.dot(
#             latents[...,self.varied_in_sampling],self.latent_bases_varied)

#     def setup_latents_bases(self):
#         """
#         Computes the latents bases vector for converting latents vectors to indices.
#         """
#         self.num_latents_varied = self.num_latents[self.varied_in_sampling]
#         self.latent_bases = np.concatenate([
#             np.cumprod(self.num_latents[::-1])[::-1][1:],[1]])
#         self.latent_bases_varied = np.concatenate([
#             np.cumprod(self.num_latents_varied[::-1])[::-1][1:],[1]])    
#         self.dataset_size = np.prod(self.num_latents_varied)

#     def transition(self, idx):
#         pass

#     def observe_n_transitions(self, idx):
#         indices = np.empty(shape=(idx.shape[-1], self.n_transitions+1), dtype=int)
#         transitions = []
#         indices[:, 0] = idx
#         for i in range(self.n_transitions):
#             idx2, dj = self.transition(indices[:,i])
#             indices[:,i+1]= idx2
#             transitions.append(dj)
#         transitions = np.stack(transitions,axis=1)
#         return indices,transitions
    
#     def get_val_batch(self):
#         pass
    

#     def resample_data(self):
#         pass
    
#     @property
#     def action_shape(self) -> list:
#         pass

#     @property
#     def in_shape(self) -> list:
#         pass
