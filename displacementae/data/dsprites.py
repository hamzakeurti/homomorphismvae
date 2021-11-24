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

class DspritesDataset(Dataset):
    def __init__(self,root='',rseed=None, fixed_in_sampling=[], 
                fixed_values=[],fixed_in_intervention=[],intervene=True,
                intervention_range=[-1,1], num_train = 200, num_val=30):
        super().__init__()

        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random

        self.num_train = num_train
        self.num_val = num_val

        self._rand = rand
        self._rseed = rseed
        self.intervene = intervene
        self.joints = np.arange(6)
        self.n_joints = 6
        self.fixed_in_sampling = fixed_in_sampling
        self.fixed_values = fixed_values
        self.varied_in_sampling = np.array([i for i in self.joints \
            if i not in self.fixed_in_sampling])
        self.fixed_in_intervention = fixed_in_intervention
        self.intervened_on = np.array([i for i in self.joints \
            if i not in self.fixed_in_intervention])
        if not self.intervene:
            self.intervened_on = np.array([])
            self.fixed_in_intervention = self.joints
        self.intervention_range = intervention_range
        
        self._root = root
        
        self._images, self._classes, self._values = self._process_hdf5()
        self.num_classes = self._classes[-1] + 1

        data = {}
        data["in_shape"] = [1,64,64]
        data["action_shape"] = [len(self.intervened_on)]
        self._data = data

        self.cumulative_product = np.concatenate([[1],np.cumprod(self.num_classes[::-1])])

        self.cum_prod_fix = [1]
        
        self.n_samples = 1
        for f,num in reversed(list(enumerate(self.num_classes))):
            if f not in self.fixed_in_sampling:
                self.n_samples *= num
                self.cum_prod_fix.append(self.n_samples)

        # TODO: Generate indices before and after intervention here
        self.all_indices = self.get_indices_vary_joints(self.varied_in_sampling)
        ### Training samples:
        self.train_idx1 = rand.choice(self.all_indices,size=num_train)
        if self.intervene:
            train_idx2_dj = [self.f_intervene(i) for i in self.train_idx1]
            self.train_idx2 = [elem[0] for elem in train_idx2_dj]
            self.train_dj = [elem[1] for elem in train_idx2_dj]
        ### Evaluation samples
        self.val_idx1 = np.sort(
            rand.choice(self.all_indices,size=num_val,replace=False))
        if self.intervene:
            val_idx2_dj = [self.f_intervene(i) for i in self.val_idx1]
            self.val_idx2 = [elem[0] for elem in val_idx2_dj]
            self.val_dj = [elem[1] for elem in val_idx2_dj]
        
    def _process_hdf5(self):
        filepath = os.path.join(self._root,'dsprites.hdf5')
        self._file = h5py.File(filepath,'r')
        images = self._file[IMGS]
        classes = self._file[LATENTS][CLASSES]
        values = self._file[LATENTS][VALUES]
        return images, classes, values

    def __len__(self):
        return self.num_train

    def __getitem__(self, idx):
        idx1 = self.train_idx1[idx]
        image1, cls1 = self._images[idx1], self._classes[idx1]
        if self.intervene:
            idx2 = self.train_idx2[idx]
            dj = self.train_dj[idx]
            image2, cls2 = self._images[idx2], self._classes[idx2]
            return image1, cls1, image2, cls2, dj
        else:
            return image1, cls1, image1, cls1, 0

    def get_images_batch(self, indices):
        imgs, labels = self._images[indices], self._classes[indices]
        imgs = np.expand_dims(imgs,axis=1)
        return imgs, labels

    def get_val_batch(self):
        idx1 = self.val_idx1
        image1 = np.expand_dims(self._images[idx1], axis=1) 
        cls1 = self._classes[idx1]
        if self.intervene:
            idx2 , dj = self.val_idx2, self.val_dj
            image2 = np.stack([self._images[i] for i in idx2],axis=0)
            image2 = np.expand_dims(image2, axis=1) 
            cls2 = np.stack([self._classes[i] for i in idx2],axis=0)
            return image1, cls1, image2, cls2, dj
        else:
            return image1, cls1, image1, cls1, 0

    def f_intervene(self,index):
        # intervention in the vicinity in the joints space 
        joints = self._classes[index]
        #sample displacement
        if self.fixed_in_intervention:
            len_dj = self.n_joints - len(self.fixed_in_intervention)
        else:
            len_dj = self.n_joints
        dj = np.zeros(self.n_joints)
        dj[self.intervened_on] = self._rand.randint(
            low=self.intervention_range[0],high=self.intervention_range[1]+1,
            size = len_dj)
        new_joints = joints
        new_joints,dj = self._intervene_linear(new_joints,dj)
        new_joints,dj = self._intervene_circular(new_joints,dj)
        i2 = self.joints_to_index(new_joints)
        return i2,dj

    def _intervene_linear(self,joints,dj):
        new_joints = joints
        lin_idx = [LatentIdx.SCALE,LatentIdx.POSX,LatentIdx.POSY]
        new_joints[lin_idx] = np.clip(joints[lin_idx] + dj[lin_idx],0,self.num_classes[lin_idx])
        dj[lin_idx] = new_joints[lin_idx] - joints[lin_idx]
        return new_joints,dj
    
    def _intervene_circular(self,joints,dj):
        rot_idx = [LatentIdx.ORIENT]
        new_joints = joints
        new_joints[rot_idx] = (joints[rot_idx] + dj[rot_idx]) % self.num_classes[rot_idx]
        return new_joints,dj

    def joints_to_index(self,joints):
        index = 0
        base = 1
        for j,joint in reversed(list(enumerate(joints))):
            index += joint * base
            base *= self.num_classes[j]
        return index

    def get_index(self,i):
        """
        Transfers indices from range (0,self.n_samples) to indices of samples 
        in the dataset with desired fixed factors.
        """
        ret = 0
        k = 0
        for f in range(len(self.num_classes)-1,-1,-1):
            if f in self.fixed_in_sampling:
                val = self.fixed_values[self.fixed_in_sampling.index(f)]
                ret += val*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) \
                    * self.cumulative_product[::-1][f+1]
                k+=1
        return ret

    def get_indices_vary_joints(self,vary_joints):
        indices = []
        joints = [0] * self.n_joints
        for j in range(len(self.fixed_in_sampling)):
            joints[self.fixed_in_sampling[j]] = self.fixed_values[j]
        for j in self.joints:
            if (j not in self.fixed_in_sampling) and (j not in vary_joints):
                # pick a random value for the joint
                joints[j] = self.num_classes[j]//2
        self.recursive_joints_population(indices,vary_joints,joints)
        return indices

    def recursive_joints_population(self, ret_indices, vary_joints, joints,
                                    curr_j=0):
        joint_index = vary_joints[curr_j]
        for val in range(self.num_classes[joint_index]):
            joints[joint_index] = val
            if curr_j == len(vary_joints)-1:
                idx = self.joints_to_index(joints)
                ret_indices.append(idx)
            else:
                self.recursive_joints_population(
                    ret_indices,vary_joints,joints,curr_j+1)

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
    
class FixedJointsSampler(Sampler):
    def __init__(self,fixed_joints,fixed_values,dataset=None,shuffle=False):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle
        self.fixed = fixed_joints
        self.vals = fixed_values
        self.facs = [None]*len(dataset.joints)
        self.num_classes = dataset.num_classes
        for i in range(len(self.fixed)):
            self.facs[self.fixed[i]] = self.vals[i]
        

        self.cumulative_product = np.concatenate([[1],np.cumprod(self.num_classes[::-1])])

        self.cum_prod_fix = [1]
        
        self.n_samples = 1
        for f,num in reversed(list(enumerate(self.num_classes))):
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
        for f in range(len(self.num_classes)-1,-1,-1):
            if self.facs[f] is not None:
                ret += self.facs[f]*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) * self.cumulative_product[::-1][f+1]
                k+=1
        return ret


if __name__ == '__main__':
    pass