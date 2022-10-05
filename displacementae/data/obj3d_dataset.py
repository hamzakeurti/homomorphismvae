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
# @title          :displacementae/data/obj3d_dataset.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :10/09/2022
# @version        :1.0
# @python_version :3.7.4
"""
Dataset of a 3D object in different orientations.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.obj3d` contains a data handler for a hdf5 dataset 
generated from .obj models.
"""

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py

import data.transition_dataset as trns_dataset




class Obj3dDataset(trns_dataset.TransitionDataset):
    def __init__(self, root, rseed=None, transitions_on=True,
                 n_transitions: int = None,
                 num_train=200,
                 num_val: int = 30,
                 resample:bool=False,
                 num_samples:int=200):
        super().__init__(rseed, transitions_on, n_transitions)

        # Read Data from file.
        self._root = root
        self.resample = resample
        self.num_samples = num_samples

        # Number of samples
        self.num_train = num_train
        self.num_val = num_val
        
        self.load_data()
        self.sample_val_batch()
        

        data = {}
        data["in_shape"] = self._images.shape[2:]
        data["action_shape"] = [self._transitions.shape[-1]]
        self.action_dim = self._transitions.shape[-1]
        self._data = data




    def load_data(self):
        """
        Loads samples from an hdf5 dataset.
        """
        filepath = os.path.join(self._root)
        if self.resample:
            self.resample_data()
        else:
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][()]
                self._transitions = f['actions'][()] 
            # images = self._file['images']
            # transitions = self._file['rotations']

    def resample_data(self):
        """
        Replaces new samples in memory.
        """
        if self.resample:
            if hasattr(self,"_images"):
                del self._images
                del self._transitions
            indices = np.sort(
                    np.random.choice(
                        self.num_train,size=self.num_samples, replace=False))
            filepath = os.path.join(self._root)
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][indices]
                self._transitions = f['actions'][indices]
        else:
            pass 
    
    def sample_val_batch(self):
        filepath = os.path.join(self._root)
        nt = self.num_train
        nv = self.num_val
        with h5py.File(filepath,'r') as f:
            n = f['images'].shape[0]
            if  n < (nt+nv):
                raise ValueError(f"Not enough samples {n} for chosen " + 
                    f"--num_train {nt} and --num_val {nv}")
            self.val_imgs = f['images'][nt:nt+nv]
            self.val_actions = f['actions'][nt:nt+nv,1:]


    def __len__(self):
        if self.resample:
            return self.num_samples
        else:
            return self.num_train


    def __getitem__(self, idx):
        images = self._images[idx]
        dj = self._transitions[idx,1:]
        return images, [], dj


    @property
    def n_actions(self):
        """
        Number of all possible discrete actions.
        """


    def get_example_actions(self):
        a = np.zeros((self.action_dim*2+1,self.action_dim))
        for i in range(self.action_dim):
            a[1+2*i:3+2*i,i] = np.array([1,-1])
        
        # if self.rotate_actions:
        #     rot_a = a.copy()
        #     rot_a[...,:2] = rot_a[...,:2] @ self._rot_mat
        #     return rot_a, a
        # else:
        return a, a

    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]

    def get_val_batch(self):
    #     imgs = self._images[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self.val_imgs, None, self.val_actions


if __name__ == '__main__':
    pass