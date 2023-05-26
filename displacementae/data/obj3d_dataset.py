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
import numpy.typing as npt
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py
from typing import Any, Generator, Tuple, Optional

from displacementae.data.transition_dataset import TransitionDataset
from displacementae.utils import misc


class Obj3dDataset(TransitionDataset):
    def __init__(self, 
                 root:str, 
                 rseed:int=None,
                 n_transitions: int = None,
                 num_train:int=200,
                 num_val: int = 30,
                 resample:bool=False,
                 num_samples:int=200,
                 normalize_actions:bool=False,
                 rollouts:bool=False,
                 rollouts_path:Optional[str]=None,
                 rollouts_batch_size:Optional[int]=None,):
        super().__init__(rseed, n_transitions)

        # Read Data from file.
        self._root = os.path.expanduser(
                    os.path.expandvars(root))
        self._resample = resample
        self._num_samples = num_samples
        self._normalize_actions = normalize_actions

        # Number of samples
        self._num_train = num_train
        self._num_val = num_val
        
        self._load_data()
        self._load_attributes()
        self._sample_val_batch()

        self._rollouts = rollouts
        if self._rollouts:
            assert rollouts_path is not None
            assert rollouts_batch_size is not None
            self._rollouts_path = os.path.expanduser(
                        os.path.expandvars(rollouts_path))
            self._rollouts_batch_size = rollouts_batch_size
            self._load_rollouts()

        
        self._rots_idx = np.array([])
        self._trans_idx = np.array([])
        self._col_idx = np.array([])
        
        k = 0
        if self._rotate:
            n_rots_act = 3
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
        # this is the input size to the group representation
        data["action_units"] = self._transitions.shape[-1]

        # This is the dimensionality of the action space
        
        self.action_dim = 3*self._rotate + 3*self._translate + self._color
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

        if self._resample:
            self.resample_data()
        else:
            with h5py.File(self._root,'r') as f:
                self._images = f['images'][:self._num_train]
                self._transitions = f['actions'][:self._num_train] 
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
            self._attributes_dict = dict(f.attrs)
        #         "obj_filename":obj_filename,  
        # "figsize":figsize,
        # "dpi":dpi, 
        # "lim":lim,
        # self._translate_only=self._attributes_dict["translate_only"]
        self._mode = self._attributes_dict["mode"] 
        self._figsize = self._attributes_dict["figsize"]
        self._n_steps=self._attributes_dict["n_steps"] 
        self._n_samples=self._attributes_dict["n_samples"]
        
        self._rotate = self._attributes_dict["rotate"]
        if self._rotate:
            self._rots_range=misc.str_to_floats(
                        self._attributes_dict["rotation_range"])
            self._rotation_format=self._attributes_dict["rotation_format"]
            if self._mode=='discrete':
                self._rots_n_values=self._attributes_dict["n_values"]

        self._color= self._attributes_dict["color"]
        if self._color:
            self._n_colors=self._attributes_dict["n_colors"]
        
        self._translate=self._attributes_dict["translate"]
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
            self._val_actions = f['actions'][nt:nt+nv]
            if self._normalize_actions:
                self._val_actions /= self._M


    def _load_rollouts(self):

        with h5py.File(self._rollouts_path,'r') as f:
            assert f.attrs['figsize'] == self._figsize
            assert f.attrs['mode'] == self._mode
            assert f.attrs['rotate'] == self._rotate
            assert f.attrs['translate'] == self._translate
            assert f.attrs['color'] == self._color
            if self._rotate:
                rng = misc.str_to_floats(f.attrs['rotation_range'])
                assert rng == self._rots_range
                assert f.attrs['rotation_format'] == self._rotation_format
            if self._color:
                assert f.attrs['n_colors'] == self._n_colors

            self._roll_imgs = f['images'][:]
            self._roll_actions = f['actions'][:] 
            if self._normalize_actions:
                self._roll_actions /= self._M

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
            else: # color
                a[1+2*i:3+2*i,i] = np.array([1,-1])

        if self._rotation_format == "mat":
            R = misc.euler_to_mat(a[:,:3]) # R shape: [n,9]
            a_in = np.concatenate([R,a[:,3:].copy()], axis=-1)
        else:
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


    def get_rollouts(self) -> Generator[
         Tuple[npt.NDArray, npt.NDArray], None, None]:
        """Returns a generator of rollouts."""
        
        if not self._rollouts:
            raise ValueError("Rollouts were not loaded.")
        
        b = self._rollouts_batch_size
        for i in range(0, self._roll_imgs.shape[0], b): 
            yield self._roll_imgs[i:i+b],\
                    self._roll_actions[i:i+b] # type: ignore
            

    def get_n_rollouts(self, n: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns the first n rollouts."""

        if not self._rollouts:
            raise ValueError("Rollouts were not loaded.")
        return self._roll_imgs[:n], \
                self._roll_actions[:n] # type: ignore
    
            
if __name__ == '__main__':
    pass