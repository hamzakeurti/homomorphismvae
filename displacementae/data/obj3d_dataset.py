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
from typing import Any, Tuple

from data.transition_dataset import TransitionDataset




class Obj3dDataset(TransitionDataset):
    def __init__(self, 
                 root:str, 
                 rseed:int=None,
                 n_transitions: int = None,
                 num_train:int=200,
                 num_val: int = 30,
                 resample:bool=False,
                 num_samples:int=200,
                 normalize_actions:bool=False):
        super().__init__(rseed, n_transitions)

        # Read Data from file.
        self._root = root
        self._resample = resample
        self._num_samples = num_samples
        self._normalize_actions = normalize_actions

        # Number of samples
        self._num_train = num_train
        self._num_val = num_val
        
        self._load_data()
        self._load_attributes()
        self._sample_val_batch()
        
        self._rots_idx = np.array([])
        self._trans_idx = np.array([])
        self._col_idx = np.array([])
        
        k = 0
        if self._rotate:
            n_rots_act = 9 if self._rotation_matrix_action else 3
            self._rots_idx = np.arange(n_rots_act)
            k += n_rots_act
        
        if self._translate:
            self._trans_idx = np.arange(start=k,stop=k+3)
            k+=3

        if self._color:
            self._col_idx = np.array(self._transitions.shape[-1]-1)        
        
        rng = self._rots_range[1] - self._rots_range[0]
        if self._mode=='continuous':
            self._rots_stepsize=rng/4
        else:
            self._rots_stepsize=rng/(self._rots_n_values-1)
        

        data = {}
        data["in_shape"] = self._images.shape[2:]
        data["action_units"] = self._transitions.shape[-1]
        data["action_dim"] = self._transitions.shape[-1]
        self.action_dim = self._transitions.shape[-1]
        self._data = data


    def __len__(self):
        if self._resample:
            return self._num_samples
        else:
            return self._num_train


    def __getitem__(self, idx):
        images = self._images[idx]
        dj = self._transitions[idx]
        return images, [], dj


    def resample_data(self) -> None:
        """
        Replaces new samples in memory.
        """
        if self._resample:
            if hasattr(self,"__images"):
                del self._images
                del self._transitions
            indices = np.sort(
                    self._rand.choice(
                        self._num_train,size=self._num_samples, replace=False))
            filepath = os.path.join(self._root)
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][indices]
                self._transitions = f['actions'][indices,1:]
                if self._normalize_actions:
                    self._M = np.abs(self._transitions).max(axis=(0,1))
                    self._transitions /= self._M
        else:
            pass 
    

    @property
    def in_shape(self):
        return self._data["in_shape"]


    @property
    def action_units(self) -> int:
        return self._data["action_units"]
    

    @property
    def n_actions(self):
        """
        Number of all possible discrete actions.
        """
        pass


    def _load_data(self):
        """
        Loads samples from an hdf5 dataset.
        """
        filepath = os.path.join(self._root)
        if self._resample:
            self.resample_data()
        else:
            with h5py.File(filepath,'r') as f:
                self._images = f['images'][:self._num_train]
                self._transitions = f['actions'][:self._num_train,1:] 
                if self._normalize_actions:
                    self._M = np.abs(self._transitions).max(axis=(0,1))
                    self._transitions /= self._M
            # images = self._file['images']
            # transitions = self._file['rotations']

    def _load_attributes(self):
        """
        Loads the atributes of the dataset
        """
        with h5py.File(self._root,'r') as f:
            self._attributes_dict = dict(f['images'].attrs)
        #         "obj_filename":obj_filename,  
        # "figsize":figsize,
        # "dpi":dpi, 
        # "lim":lim,
        # self._translate_only=self._attributes_dict["translate_only"]
        self._mode = self._attributes_dict["mode"] 
        self._translate=self._attributes_dict["translate"]
        self._rotate = self._attributes_dict["rotate"]
        self._color= self._attributes_dict["color"]
        self._rots_range=self._attributes_dict["rots_range"]
        self._n_steps=self._attributes_dict["n_steps"] 
        self._n_samples=self._attributes_dict["n_samples"]
        self._rots_n_values=self._attributes_dict["n_values"]
        self._rotation_matrix_action=self._attributes_dict["rotation_matrix_action"] 
        if self._translate:
            self._trans_grid=self._attributes_dict["translation_grid"]
            self._trans_stepsize=self._attributes_dict["translation_stepsize"]
            self._trans_range=self._attributes_dict["translation_range"]
        

    def _sample_val_batch(self):
        filepath = os.path.join(self._root)
        nt = self._num_train
        nv = self._num_val
        with h5py.File(filepath,'r') as f:
            n = f['images'].shape[0]
            if  n < (nt+nv):
                raise ValueError(f"Not enough samples {n} for chosen " + 
                    f"--num_train={nt} and --num_val={nv}")
            self._val_imgs = f['images'][nt:nt+nv]
            self._val_actions = f['actions'][nt:nt+nv,1:]
            if self._normalize_actions:
                self._val_actions /= self._M




    def get_example_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """returns a set of example actions (transition signals) with labels. 

        :return: a tuple of a batch of action signals as perceived by the agent 
                 and associated labels.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        a = np.zeros((self.action_dim*2+1,self.action_dim))
        for i in range(self.action_dim):
            if i in self._rots_idx:
                a[1+2*i:3+2*i,i] = np.array([1,-1])*self._rots_stepsize
            elif i in self._trans_idx:
                a[1+2*i:3+2*i,i] = np.array([1,-1])*self._trans_stepsize
            else:
                a[1+2*i:3+2*i,i] = np.array([1,-1])
            a_in = a.copy()
            if self._normalize_actions:
                a_in /= self._M
        return a_in, a

    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of evaluation samples.

        :return: a tuple of a batch of observation evaluation samples and 
                 transition evaluation samples. 
        :rtype: Tuple[np.ndarray, np.ndarray]
        """    #     imgs = self._images[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self._val_imgs, None, self._val_actions


if __name__ == '__main__':
    pass