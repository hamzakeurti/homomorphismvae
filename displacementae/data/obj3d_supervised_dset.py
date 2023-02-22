#!/usr/bin/env python3
# Copyright 2023 Hamza Keurti
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
# @title          :displacementae/data/obj3d_supervised_dset.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :20/02/2022
# @version        :1.0
# @python_version :3.7.4
"""
Dataset of a 3D object in different poses (orientations/positions/colors).
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A data handler for the supervised setting. A similar handler for the transition 
setting is provided in :mod:`data.obj3d_dataset`.
This handler loads an hdf5 dataset of images and labels previously generated 
using the :mod:`data.obj3d` module. 
"""

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py
from typing import Any, Tuple




class Obj3dSupervisedDataset(Dataset):
    def __init__(self, 
                 root:str, 
                 rseed:int=None,
                 num_train:int=200,
                 num_val:int=30,
                 ):
        super().__init__()
        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random
        self._rand = rand
        self._rseed = rseed

        # Read Data from file.
        self._root = root

        # Number of samples
        self._num_train = num_train
        self._num_val = num_val

        self._load_data()
        self._load_attributes()
        
        self._rots_idx = np.array([])
        self._trans_idx = np.array([])
        self._col_idx = np.array([])
        
        nrots = 9 if self._rotation_matrix_action else 3
        ntrans = 3
        ncols = 1
        n = 0
        if self._rotate:
            self._rots_idx = np.arange(nrots)
            n += nrots
        
        if self._translate:
            self._trans_idx = np.arange(start=n,stop=n+ntrans)
            n+= ntrans
        if self._color:
            self._col_idx = np.array(n-1)        
        
        rng = self._rots_range[1] - self._rots_range[0]
        if self._mode=='continuous':
            self._rots_stepsize=rng/4
        else:
            self._rots_stepsize=rng/(self._rots_n_values-1)
        

        self._labels = self._actions
        self._val_labels = self._val_actions

        data = {}
        data["in_shape"] = self._imgs.shape[2:]
        data["action_units"] = self._labels.shape[-1]
        self._data = data


    def __len__(self):
        if self._resample:
            return self._num_samples
        else:
            return self._num_train


    def __getitem__(self, idx):
        images = self._imgs[idx]
        labels = self._labels[idx]
        return images, labels
    

    @property
    def in_shape(self):
        return self._data["in_shape"]


    @property
    def action_units(self) -> int:
        return self._data["action_units"]


    def _load_data(self):
        """
        Loads samples from an hdf5 dataset.
        """
        filepath = os.path.join(self._root)

        nt = self._num_train
        nv = self._num_val
        with h5py.File(filepath,'r') as f:
            self._imgs = f['images'][:nt]
            self._actions = f['actions'][:nt]
            self._rot_mats = f['positions'][:nt]
            
            n = f['images'].shape[0]
            if  n < (nt+nv):
                raise ValueError(f"Not enough samples {n} for chosen " + 
                    f"--num_train={nt} and --num_val={nv}")
            
            self._val_imgs = f['images'][nt:nt+nv]
            self._val_actions = f['actions'][nt:nt+nv]
            self._val_rots_mats = f['positions'][nt:nt+nv]


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
        self._mode = self._attributes_dict["mode"] 
        self._translate=self._attributes_dict["translate"]
        self._rotate=self._attributes_dict["rotate"]
        self._rotation_matrix_action = self._attributes_dict["rotation_matrix_action"]
        self._rots_range=self._attributes_dict["rots_range"]
        self._n_steps=self._attributes_dict["n_steps"] 
        self._n_samples=self._attributes_dict["n_samples"]
        self._color= self._attributes_dict["color"]
        self._rots_n_values=self._attributes_dict["n_values"] 
        if self._translate:
            self._trans_grid=self._attributes_dict["translation_grid"]
            self._trans_stepsize=self._attributes_dict["translation_stepsize"]
            self._trans_range=self._attributes_dict["translation_range"]


    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of evaluation samples.

        :return: a tuple of a batch of observation evaluation samples and 
                 the batch of their associated labels. 
        :rtype: Tuple[np.ndarray, np.ndarray]
        """    #     imgs = self._imgs[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self._val_imgs, self._val_labels


if __name__ == '__main__':
    # pass

    from data.obj3d_supervised_dset import Obj3dSupervisedDataset

    root = 'C:/Users/hamza/datasets/obj3d/collect/bunny1.hdf5'
    dataset = Obj3dSupervisedDataset(root=root,rseed=5,use_rotation_matrix=True,num_train=1,num_val=0)
    x = dataset[0]
    print(x.shape)