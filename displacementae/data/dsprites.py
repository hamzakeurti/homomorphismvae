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
Dataset of simple shapes in different positions.
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The module :mod:`data.dsprites` contains a data handler for the 
`dsprites dataset <https://github.com/deepmind/dsprites-dataset>`.
"""

import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,Sampler
import os
import h5py

IMGS = "imgs"
LATENTS = "latents"
CLASSES = "classes"
VALUES = "values"

class LatentIdx:
    COLOR = 0
    SHAPE = 1
    SCALE = 2 
    ORIENT = 3
    POSX = 4
    POSY = 5

LATENT_NAMES = ['color', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y']

class DspritesDataset(Dataset):
    def __init__(self,root,rseed=None, fixed_in_sampling=[], 
                fixed_values=[], fixed_in_action=[], transitions_on=True,
                action_range=[-1,1], num_train = 200, num_val=30,
                cyclic_trans=False):
        super().__init__()

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

        # latents config
        self.transitions_on = transitions_on
        self.n_joints = 6
        self.joints = np.arange(self.n_joints)
        
        self.fixed_in_sampling = fixed_in_sampling
        self.fixed_values = fixed_values
        self.varied_in_sampling = [i for i in self.joints \
            if i not in self.fixed_in_sampling]
        self.fixed_in_action = fixed_in_action
        self.varied_in_action = np.array([i for i in self.joints \
            if i not in self.fixed_in_action])
        if not self.transitions_on:
            self.varied_in_action = np.array([])
            self.fixed_in_action = self.joints
        self.transition_range = action_range
        # Types of latents
        if not cyclic_trans:
            self.lin_idx = [LatentIdx.SCALE,LatentIdx.POSX,LatentIdx.POSY]
            self.rot_idx = [LatentIdx.ORIENT]
        else:
            self.lin_idx = [LatentIdx.SCALE]
            self.rot_idx = [LatentIdx.ORIENT,LatentIdx.POSX,LatentIdx.POSY]

        # Read Data from file.
        self._root = root
        self._images, self._classes, self._values = self._process_hdf5()
        
        # Number of values for each latent
        self.num_latents = self._classes[-1] + 1 
        self.num_latents_varied = self.num_latents[self.varied_in_sampling]
        self.latent_bases = np.concatenate([
            np.cumprod(self.num_latents[::-1])[::-1][1:],[1]])
        self.latent_bases_varied = np.concatenate([
            np.cumprod(self.num_latents_varied[::-1])[::-1][1:],[1]])    
        self.dataset_size = np.prod(self.num_latents_varied)

        data = {}
        data["in_shape"] = [1,64,64]
        data["action_shape"] = [len(self.varied_in_action)]
        self._data = data

        # Get Dataset subset corresponding to fixed_in_sampling constraint.
        self.all_indices = self._get_subset_indices()
        self.images = np.expand_dims(self._images[self.all_indices],1)
        self.latents = self._classes[self.all_indices]

        ### Training samples:
        self.train_idx1 = rand.choice(self.dataset_size,size=num_train)
        if self.transitions_on:
            self.train_idx2, self.train_dj = self.f_intervene(self.train_idx1)
        ### Evaluation samples
        self.val_idx1 = rand.choice(self.dataset_size, 
                                    size=num_val,replace=False)
        if self.transitions_on:
            self.val_idx2, self.val_dj = self.f_intervene(self.val_idx1)
        
    def _process_hdf5(self):
        """
        opens the hdf5 dataset file.
        """
        filepath = os.path.join(self._root,'dsprites.hdf5')
        self._file = h5py.File(filepath,'r')
        images = self._file[IMGS]
        classes = self._file[LATENTS][CLASSES]
        values = self._file[LATENTS][VALUES]
        return images, classes, values

    def _get_subset_indices(self):
        """
        Generate a list of indices in the dsprites dataset corresponding to 
        the subset of configurations corresponding to the fixed_latents. 
        """
        # specify all values that can be taken by each latent
        latents_spans = [np.arange(self.num_latents[i]) \
            if i not in self.fixed_in_sampling 
            else np.array(self.fixed_values[self.fixed_in_sampling.index(i)]) 
            for i in range(self.n_joints)]
        mesh = np.meshgrid(*latents_spans)
        mesh = [mesh[i].reshape(-1) for i in range(self.n_joints)]
        all_latents = np.array([[mesh[i][j] for i in range(self.n_joints)] \
            for j in range(len(mesh[0]))])
        indices = np.dot(all_latents,self.latent_bases)
        return indices

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        idx1 = self.train_idx1[idx]
        image1 = self.images[idx1]
        latents1 = self.latents[idx1]
        if self.transitions_on:
            idx2 = self.train_idx2[idx]
            dj = self.train_dj[idx]
            image2 = self.images[idx2]
            latents2 = self.latents[idx2]
            return image1, latents1, image2, latents2, dj
        else:
            return image1, latents1, image1, latents1, 0

    def get_latent_name(self,id):
        """
        Returns the name of the latent corresponding to the input id.
        """
        return LATENT_NAMES[id]

    def get_images_batch(self, indices):
        """
        Returns a batch of images and labels corresponding to input indices.
        """
        imgs, labels = self._images[indices], self._classes[indices]
        imgs = np.expand_dims(imgs,axis=1)
        return imgs, labels

    def get_val_batch(self):
        idx1 = self.val_idx1
        image1 = self.images[idx1]
        latents1 = self.latents[idx1]
        if self.transitions_on:
            idx2 = self.val_idx2
            dj = self.val_dj
            image2 = self.images[idx2]
            latents2 = self.latents[idx2]
            return image1, latents1, image2, latents2, dj
        else:
            return image1, latents1, image1, latents1, 0

    def f_intervene(self,index):
        """"""
        joints = self.latents[index]
        #sample displacement
        if self.fixed_in_action:
            len_dj = self.n_joints - len(self.fixed_in_action)
        else:
            len_dj = self.n_joints
        dj = np.zeros((joints.shape[0],self.n_joints)).squeeze()
        dj[...,self.varied_in_action] = self._rand.randint(
            low=self.transition_range[0],high=self.transition_range[1]+1,
            size = (joints.shape[0],len_dj))
        new_joints,dj = self._intervene_linear(joints,dj)
        new_joints,dj = self._intervene_circular(new_joints,dj)
        indices2 = self.joints_2_index(new_joints)
        return indices2,dj


    def _intervene_linear(self,joints,dj):
        new_joints = joints.copy()
        lin_idx = self.lin_idx
        new_joints[...,lin_idx] = np.clip(
            joints[...,lin_idx] + dj[...,lin_idx],0,self.num_latents[lin_idx]-1)
        dj[...,lin_idx] = new_joints[...,lin_idx] - joints[...,lin_idx]
        return new_joints,dj
    
    def _intervene_circular(self,joints,dj):
        """
        Adds a displacement on latents with a cyclic topology.


        """
        rot_idx = self.rot_idx
        num_latents = self.num_latents[rot_idx]
        # Last coincides with first 
        num_latents[0] = num_latents[0] - 1
        new_joints = joints.copy()
        new_joints[...,rot_idx] = (joints[...,rot_idx] + dj[...,rot_idx])\
             % num_latents
        return new_joints,dj

    def joints_to_index(self,joints):
        index = 0
        base = 1
        for j,joint in reversed(list(enumerate(joints))):
            index += joint * base
            base *= self.num_latents[j]
        return index

    def joints_2_index(self,joints):
        return np.dot(
            joints[...,self.varied_in_sampling],self.latent_bases_varied)

    def get_index(self,i):
        """
        Transfers indices from range (0,self.n_samples) to indices of samples 
        in the dataset with desired fixed factors.
        """
        ret = 0
        k = 0
        for f in range(len(self.num_latents)-1,-1,-1):
            if f in self.fixed_in_sampling:
                val = self.fixed_values[self.fixed_in_sampling.index(f)]
                ret += val*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) \
                    * self.cumulative_product[::-1][f+1]
                k+=1
        return ret

    def get_indices_vary_latents(self,vary_latents):
        indices = []
        assert np.array([j in self.varied_in_sampling for j in vary_latents]).all()

        latents_spans = []
        for i in range(self.n_joints):
            if i in vary_latents:
                latents_spans.append(np.arange(self.num_latents[i]))
            elif i in self.varied_in_sampling:
                latents_spans.append(self.num_latents[i]//2)

        mesh = np.meshgrid(*latents_spans)
        mesh = [m.reshape(-1) for m in mesh]
        all_latents = np.array([[m[j] for m in mesh] \
            for j in range(len(mesh[0]))])
        
        indices = np.dot(all_latents,self.latent_bases_varied)
        return indices

    @property
    def allowed_indices(self):
        if hasattr(self,'_allowed_indices'):
            pass
        else:
            self._allowed_indices = [self.get_index(i) for i in range(self.n_samples)]
        return self._allowed_indices

    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]
    
    # def f_intervene(self,index):
    #     # intervention in the vicinity in the joints space 
    #     joints = self._classes[index]
    #     #sample displacement
    #     if self.fixed_in_intervention:
    #         len_dj = self.n_joints - len(self.fixed_in_intervention)
    #     else:
    #         len_dj = self.n_joints
    #     dj = np.zeros(self.n_joints)
    #     dj[self.intervened_on] = self._rand.randint(
    #         low=self.intervention_range[0],high=self.intervention_range[1]+1,
    #         size = len_dj)
    #     new_joints = joints
    #     new_joints,dj = self._intervene_linear(new_joints,dj)
    #     new_joints,dj = self._intervene_circular(new_joints,dj)
    #     i2 = self.joints_to_index(new_joints)
    #     return i2,dj


class FixedJointsSampler(Sampler):
    def __init__(self,fixed_joints,fixed_values,dataset=None,shuffle=False):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle
        self.fixed = fixed_joints
        self.vals = fixed_values
        self.facs = [None]*len(dataset.joints)
        self.num_latents = dataset.num_latents
        for i in range(len(self.fixed)):
            self.facs[self.fixed[i]] = self.vals[i]
        

        self.cumulative_product = np.concatenate([[1],np.cumprod(self.num_latents[::-1])])

        self.cum_prod_fix = [1]
        
        self.n_samples = 1
        for f,num in reversed(list(enumerate(self.num_latents))):
            if f not in self.fixed:
                self.n_samples *= num
                self.cum_prod_fix.append(self.n_samples)
        

    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        if not self.shuffle:
            return (self.get_index(i) for i in range(self.n_samples))
        else:
            import random
            return (self.get_index(i) for i in random.sample(list(range(self.n_samples)),self.n_samples))

    def get_index(self,i):
        """
        Transfers indices from range (0,self.n_samples) to indices of samples in the dataset with desired fixed factors.
        """
        ret = 0
        k = 0
        for f in range(len(self.num_latents)-1,-1,-1):
            if self.facs[f] is not None:
                ret += self.facs[f]*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) * self.cumulative_product[::-1][f+1]
                k+=1
        return ret


if __name__ == '__main__':
    pass