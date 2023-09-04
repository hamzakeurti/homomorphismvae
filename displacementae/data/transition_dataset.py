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

from typing import Tuple, List, Generator, Optional

import numpy as np
import numpy.typing as npt
from torch.utils.data import Dataset

from displacementae.utils import plotting_utils as plt_utils


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


    def __init__(self, rseed:Optional[int]=None, n_transitions:int=1):
        
        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random
        self._rand = rand
        self._rseed = rseed

        # Number of transitions
        self._n_transitions = n_transitions
    

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
        """
        resamples the training dataset. (Does nothing for some datasets).
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
        raise NotImplementedError
    

    def get_rollouts(self) -> Generator[
            Tuple[npt.NDArray, npt.NDArray], None, None]:
        raise NotImplementedError


    def get_n_rollouts(self, n:int) -> Tuple[npt.NDArray, npt.NDArray]:
        raise NotImplementedError


    @property
    def action_units(self) -> int:
        """Number of action units 

        :return: Dimension of the action vector.
        :rtype: int
        """
        raise NotImplementedError

    @property
    def in_shape(self) -> List[int]:
        """The shape of the observations :math:`o_t`.

        :return: A list of the dimensions of an observation sample.
                 Also contains number of channels.
                 For instance for levels of gray images, 
                 this returns `[1, height, width]`. 
        :rtype: List[int]
        """
        raise NotImplementedError


    @property
    def num_train(self) -> int:
        """Number of training samples

        :return: `int` indicating total number fo training samples.
        :rtype: int
        """
        raise NotImplementedError


    @property
    def num_val(self) -> int:
        """Number of evaluation samples

        :return: an integer indicating the number of evaluation samples.
        :rtype: int
        """
        raise NotImplementedError

    # ----------------------
    # Plotting methods
    # ----------------------

    def plot_n_step_reconstruction(self, nets, config, 
                                   device, logger, epoch, figdir) -> None:
        """
        Plots the first few transitions in the evaluation batch.

        This method saves the figure in the `figname` path,
        and logs it to WandB as well.
        """
        raise NotImplementedError


    def plot_manifold(
                    self,nets,shared, config,
                    device, logger, mode, epoch, figdir) -> None:
        """
        Plots the learned representation manifold of the dataset.

        Plots the learned representation manifold 
        (or its projection along specified representation units) 
        of the dataset (or a subset of it).
        
        To be implemented by inheriting class.
        """
        raise NotImplementedError


    def plot_manifold_pca(self, nets, shared, config,
                          device, logger, mode, epoch, figdir) -> None:
        """
        Plots the PCA projection of the learned representation manifold.

        Plots the PCA projection of the learned representation manifold 
        of the dataset or a subset of it.
        
        To be implemented by inheriting class.
        """
        raise NotImplementedError
    

    def plot_rollout_reconstruction(self, nets, config, device, logger, epoch, figdir) -> None:
        """
        Plots the reconstructions of the first few rollouts.
        """
        raise NotImplementedError



if __name__ == "__main__":
    pass