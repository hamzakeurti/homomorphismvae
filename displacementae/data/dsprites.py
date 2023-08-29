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
from sklearn.random_projection import GaussianRandomProjection
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
import os
import h5py
from typing import Tuple, List


from displacementae.data.transition_dataset import TransitionDataset
from displacementae.utils import misc
import displacementae.utils.plotting_utils as plt_utils

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


class DspritesDataset(TransitionDataset):
    def __init__(self, 
            root:str, 
            rseed:int=None, 
            fixed_in_sampling:List[int]=[],
            fixed_values:List[int]=[], 
            fixed_in_action:List[int]=[], 
            n_transitions: int = None, 
            action_range: list = [-1, 1],
            num_train=200,
            num_val: int = 30, 
            cyclic_trans: bool = False,
            dist: str = 'uniform',
            return_integer_actions: bool = False,
            rotate_actions: float = 0,
            normalize_actions:bool=True,
            plot_vary_latents:List[List[int]]=[],
            plot_repr_units:List[List[int]]=[],):
        super().__init__(rseed, n_transitions)

        # Distribution
        self.dist = dist
        self.return_integer_actions = return_integer_actions

        # Number of samples
        self._num_train = num_train
        self._num_val = num_val


        # latents config
        self.normalize_actions = normalize_actions
        self.n_latents = 6
        self.latents = np.arange(self.n_latents)

        self.fixed_in_sampling = fixed_in_sampling
        self.fixed_values = fixed_values
        self.varied_in_sampling = [i for i in self.latents
                                   if i not in self.fixed_in_sampling]
        self.fixed_in_action = fixed_in_action
        self.varied_in_action = np.array([i for i in self.latents
                                          if i not in self.fixed_in_action])
        self.transition_range = action_range
        # Types of latents
        if not cyclic_trans:
            self.lin_idx = [LatentIdx.SCALE, LatentIdx.POSX, LatentIdx.POSY]
            self.rot_idx = [LatentIdx.ORIENT]
        else:
            self.lin_idx = [LatentIdx.SCALE]
            self.rot_idx = [LatentIdx.ORIENT, LatentIdx.POSX, LatentIdx.POSY]

        # Read Data from file.
        self._root = root
        self._images, self._classes, self._values = self._process_hdf5()
        
        # Number of values for each latent
        self.num_latents = self._classes[-1] + 1 
        self._setup_latents_bases()

        #number of unit actions
        if len(self.fixed_in_action)>0:
            self.n_varied = self.n_latents - len(self.fixed_in_action)
        else:
            self.n_varied = self.n_latents

        data = {}
        data["in_shape"] = [1,64,64]
        if self.return_integer_actions:
            data["action_units"] = 1
        else:
            data["action_units"] = len(self.varied_in_action)
        data["action_dim"] = len(self.varied_in_action)

        self._data = data

        # Get Dataset subset corresponding to fixed_in_sampling constraint.
        self.all_indices = self._get_subset_indices()
        self.images = np.expand_dims(self._images[self.all_indices],1)
        self.latents = self._classes[self.all_indices]


        ### Training samples:
        self.train_start_idx = self._rand.choice(self.dataset_size,
                                                 size=num_train)
        self.train_idx, self.train_dj = self.observe_n_transitions(
                                                self.train_start_idx)
        ### Evaluation samples
        self.val_start_idx = self._rand.choice(self.dataset_size, 
                                               size=num_val, replace=False)
        self.val_idx, self.val_dj = self.observe_n_transitions(
                                               self.val_start_idx)

        if self.normalize_actions:
            self.M = np.abs(self.train_dj).max(axis=(0,1))
            self.train_dj /= self.M
            self.val_dj /= self.M
        
        if self.action_units>=2:
            self.rotate_actions = rotate_actions
        else:
            self.rotate_actions = 0


        if self.rotate_actions:
            phi = np.radians(self.rotate_actions)
            self._rot_mat = np.array([
                                        [np.cos(phi), -np.sin(phi)],
                                        [np.sin(phi), np.cos(phi)]])

            self.train_dj[...,:2] = self.train_dj[...,:2] @ self._rot_mat
            self.val_dj[...,:2] = self.val_dj[...,:2] @ self._rot_mat

        
        # Plotting config
        self.plot_vary_latents = plot_vary_latents
        self.plot_repr_units = plot_repr_units

    def __len__(self):
        return self._num_train

    def __getitem__(self, idx):
        indices = self.train_idx[idx]
        images = self.images[indices]
        latents = self.latents[indices]
        dj = self.train_dj[idx]
        return images, latents, dj

    @property
    def n_actions(self):
        """
        Number of all possible discrete actions.
        """
        if self.dist == 'uniform':
            return (self.transition_range[1] - self.transition_range[0]+1)\
                    ** self.action_dim
        if self.dist == 'disentangled':
            return (self.transition_range[1] - self.transition_range[0]+1)\
                    * self.action_dim

    @property
    def in_shape(self):
        return self._data["in_shape"]

    
    @property
    def action_units(self) -> int:
        return self._data["action_units"]
    

    @property
    def action_dim(self) -> int:
        return self._data["action_dim"]


    def _latents_2_index(self,latents:np.ndarray)->np.ndarray:
        """
        Converts a vector of latents values to its index in the subdataset.
        """
        return np.dot(
            latents[...,self.varied_in_sampling],self.latent_bases_varied)

    def _setup_latents_bases(self):
        """
        Computes the latents bases vector for converting latents vectors to indices.
        """
        self.num_latents_varied = self.num_latents[self.varied_in_sampling]
        self.latent_bases = np.concatenate([
            np.cumprod(self.num_latents[::-1])[::-1][1:],[1]])
        self.latent_bases_varied = np.concatenate([
            np.cumprod(self.num_latents_varied[::-1])[::-1][1:],[1]])    
        self.dataset_size = np.prod(self.num_latents_varied)



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
            for i in range(self.n_latents)]
        mesh = np.meshgrid(*latents_spans)
        mesh = [mesh[i].reshape(-1) for i in range(self.n_latents)]
        all_latents = np.array([[mesh[i][j] for i in range(self.n_latents)] \
            for j in range(len(mesh[0]))])
        indices = np.dot(all_latents,self.latent_bases)
        return indices


    def get_latent_name(self,id):
        """
        Returns the name of the latent corresponding to the input id.
        """
        return LATENT_NAMES[id]

    def observe_n_transitions(self, idx):
        indices = np.empty(shape=(idx.shape[-1], self._n_transitions+1), dtype=int)
        transitions = []
        indices[:, 0] = idx
        for i in range(self._n_transitions):
            idx2, dj = self._transition(indices[:,i])
            indices[:,i+1]= idx2
            transitions.append(dj)
        transitions = np.stack(transitions,axis=1)
        return indices,transitions

    def get_images_batch(self, indices):
        """
        Returns a batch of images and labels corresponding to input indices.
        """
        imgs, labels = self._images[indices], self._classes[indices]
        imgs = np.expand_dims(imgs,axis=1)
        return imgs, labels


    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of evaluation samples.

        :return: a tuple of a batch of observation evaluation samples and 
                 transition evaluation samples. 
        :rtype: Tuple[np.ndarray, , np.ndarray, np.ndarray]
        """
        indices = self.val_idx
        images = self.images[indices]
        latents = self.latents[indices]
        dj = self.val_dj
        return images, latents, dj

    def _transition(self, index):
        """"""
        latents = self.latents[index]
        if len(self.varied_in_action)==0:
            return index,None
        #sample displacement
        dj = np.zeros((latents.shape[0], self.n_latents)).squeeze()
        dj[..., self.varied_in_action] = self._sample_displacement(
            self.transition_range, self.action_dim, latents.shape[0],
            dist=self.dist)
        new_latents, dj = self._transition_linear(latents, dj)
        new_latents, dj = self._transition_circular(new_latents, dj)
        indices2 = self._latents_2_index(new_latents)
        dj = dj[..., self.varied_in_action]
        if self.return_integer_actions:
            dj = self.transition_to_index(dj)
        return indices2, dj

    def _sample_displacement(self, range, dim, n_samples, dist='uniform'):
        """Sample displacements around initial latent vector.

        Args:
            range, list: Lower and upper bound of displacement values.
            n_samples, int: Number of samples.
            dim, int: Dimensionality of the displacement vector.
            dist, str: Distribution choice to sample from, defaults to 'uniform'

        Returns:
            ndarray: displacement vector.
        """
        if dist == 'uniform':
            d = self._rand.randint(low=range[0], high=range[1]+1,
                                   size=(n_samples, dim))
        elif dist == 'disentangled':
            eye = np.eye(dim)
            # Random one hot vectors
            mask = eye[self._rand.randint(dim, size=n_samples)]
            d = mask * self._rand.randint(low=range[0], high=range[1]+1,
                                          size=(n_samples, 1))
        return d

    def _transition_linear(self, joints, dj):
        new_joints = joints.copy()
        lin_idx = self.lin_idx
        new_joints[...,lin_idx] = np.clip(
            joints[...,lin_idx] + dj[...,lin_idx],0,self.num_latents[lin_idx]-1)
        dj[...,lin_idx] = new_joints[...,lin_idx] - joints[...,lin_idx]
        return new_joints,dj

    def _transition_circular(self,joints,dj):
        """
        Adds a displacement on latents with a cyclic topology.


        """
        rot_idx = self.rot_idx
        num_latents = self.num_latents[rot_idx]
        # Last coincides with first
        num_latents[0] = num_latents[0] - 1
        new_joints = joints.copy()
        new_joints[..., rot_idx] = (joints[..., rot_idx] + dj[..., rot_idx])\
             % num_latents
        return new_joints, dj

    # def joints_to_index(self,joints):
    #     index = 0
    #     base = 1
    #     for j,joint in reversed(list(enumerate(joints))):
    #         index += joint * base
    #         base *= self.num_latents[j]
    #     return index


    def get_index(self, i):
        """
        Transfers indices from range (0,self.n_samples) to indices of samples
        in the dataset with desired fixed factors.
        """
        ret = 0
        k = 0
        for f in range(len(self.num_latents)-1, -1, -1):
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
        for i in range(self.n_latents):
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

    def transition_to_index(self, a):
        rng = self.transition_range
        n = rng[1] - rng[0]
        if self.dist == 'uniform':
            bases = n**np.arange(self.action_dim)[::-1]
            return (a - rng[0]) @ bases
        elif self.dist == 'disentangled':
            bases_p = np.arange(self.action_dim)[::-1]*rng[1]
            bases_n = np.arange(self.action_dim)[::-1]*rng[0]+\
                      self.action_dim * rng[1]
            idx = ((a>0)*bases_p + np.maximum(a,0)).sum(axis=1)
            idx += ((a<0)*bases_n + np.maximum(-a,0)).sum(axis=1)
            return idx

    def index_to_transition(self,idx):
        pass

    

    def get_example_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        a = np.zeros((self.action_dim*2+1,self.action_dim))
        for i in range(self.action_dim):
            a[1+2*i:3+2*i,i] = np.array([1,-1])
        
        if self.return_integer_actions:
            a_in = self.transition_to_index(a)
        elif self.rotate_actions:
            a_in = a.copy()
            a_in[...,:2] = a_in[...,:2] @ self._rot_mat
        else:
            a_in = a.copy()
        if self.normalize_actions:
            a_in /=self.M
        return a_in, a

    
    # -------------------
    # Plotting
    # -------------------
    def plot_manifold(
                    self, nets, shared, config,
                    device, logger, mode, epoch, figdir)->None:
        """
        Plots the learned representation manifold of the dataset.

        Plots the learned representation manifold 
        (or its projection along specified representation units) 
        of the dataset (or a subset of it).
        """
        n_repr_units = nets.n_repr_units
        for plot_units,vary_latents in zip(self.plot_repr_units,
                                           self.plot_vary_latents):             
            if max(plot_units) >= n_repr_units:
                raise ValueError(
                    "Requested plotting a representational unit which index: "+
                    f"{max(plot_units)} is too large for the "+
                    f"number of representational units: {n_repr_units}")

            indices = self.get_indices_vary_latents(vary_latents)
            latents = self.latents[indices][:,vary_latents] # type: ignore
            batch_size = config.batch_size
            n_batches = len(indices) // batch_size + 1
            
            results = []

            for i in range(n_batches):
                batch_indices = indices[ i * batch_size : (i+1) * batch_size]
                images = self.images[batch_indices]
                X = torch.FloatTensor(images).to(device)
                with torch.no_grad():
                    h, mu, logvar = nets.encode(X)
                    h = nets.normalize_representation(h)
                    results.append(h[:,plot_units].cpu().numpy())
            results = np.vstack(results).squeeze()
            
            for i in range(len(vary_latents)):
                latent = vary_latents[i]
                latent_name = self.get_latent_name(latent)

                figname = f'{epoch} - repr_manifold'
                figname += '_repr_units='+ misc.ints_to_str(plot_units)
                figname += '_varied='+ misc.ints_to_str(vary_latents)
                figname += '_true='+ misc.ints_to_str(latent) + '.pdf'

                fig_path = os.path.join(figdir, figname)

                plt_utils.plot_manifold(
                    representations=results, true_latents=latents[:,i], 
                    logger=logger, label=f'latent {latent} ({latent_name})',
                    plot_on_black=config.plot_on_black, 
                    log_wandb=config.log_wandb, path=fig_path, savefig=True)



    def plot_manifold_pca(self, nets, shared, config,
                          device, logger, mode, epoch, figdir):
        """
        Plots the PCA projection of the learned representation manifold.

        Plots the PCA projection of the learned representation manifold 
        of the dataset or a subset of it.
        
        To be implemented by inheriting class.
        """
        for vary_latents in self.plot_vary_latents:

            indices = self.get_indices_vary_latents(vary_latents)
            latents = self.latents[indices][:,vary_latents]
            batch_size = config.batch_size
            n_batches = len(indices) // batch_size + 1
            
            results = []

            for i in range(n_batches):
                batch_indices = indices[ i * batch_size : (i+1) * batch_size]
                images = self.images[batch_indices]
                X = torch.FloatTensor(images).to(device)
                with torch.no_grad():
                    h, mu, logvar = nets.encode(X)
                    h = nets.normalize_representation(h)
                    results.append(h[:,:].cpu().numpy())
            results = np.vstack(results).squeeze()

            # PCA Projection
            pca = GaussianRandomProjection(n_components=2)
            results2d = pca.fit_transform(results)

            # The loop is only to pick which latent to color and
            # which to use as marker.
            for i in range(len(vary_latents)):
                    
                latent = vary_latents[i]
                figname = f'{epoch} - repr_manifold_pca'
                figname += '_varied='+ misc.ints_to_str(vary_latents)
                figname += '_true='+ misc.ints_to_str(latent) + '.pdf'

                fig_path = os.path.join(figdir, figname)

                plt_utils.plot_manifold_markers(
                        representations=results2d, latents_clr=latents[:,i],
                        latents_mrk=latents[:,(i+1)%2], logger=logger, 
                        plot_on_black=config.plot_on_black,
                        log_wandb=config.log_wandb, path=fig_path, 
                        savefig=True,)


if __name__ == '__main__':
    pass
