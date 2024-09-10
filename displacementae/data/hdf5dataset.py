"""
Generic dataset handler for hdf5 datasets containing transitions.
"""

import h5py
import numpy as np
import numpy.typing as npt
import os
from typing import Optional, Tuple, Generator

from displacementae.data.transition_dataset import TransitionDataset
from displacementae.utils import plotting_utils as plt_utils

OBSERVATIONS = 'observations'
ACTIONS = 'actions'
TRANSITIONS = 'transitions'
ROLLOUTS = 'rollouts'



class HDF5Dataset(TransitionDataset):
    """
    Dataset handler for hdf5 datasets containing transitions of the form:
    
    - transitions: This group contains data for training.
        - observations
        - actions
    - rollouts: This group contains data for rollouts evaluation.
        - observations
        - actions
    with attributes:
    - observations_shape
    - actions_shape
    """
    def __init__(self, rseed:Optional[int]=None, n_transitions:int=1, 
                 root:Optional[str]=None,
                 num_train:Optional[int]=None,
                 num_val:Optional[int]=None,
                 normalize_actions:bool=False,
                 rollouts:bool=False,
                 rollouts_path:Optional[str]=None,
                 num_rollouts:Optional[int]=None,
                 rollouts_batch_size:Optional[int]=None,):
        super().__init__(rseed, n_transitions)
        assert root is not None
        self._root = os.path.expanduser(os.path.expandvars(root))
        
        # Number of training and validation samples.
        self._num_train = num_train
        self._num_val = num_val

        self._normalize_actions = normalize_actions

        # Load data
        self._load_data()


        # Rollouts
        self._rollouts = rollouts
        if self._rollouts:
            # Rollouts can be stored in the same file as the training data 
            # or in a separate file.
            if rollouts_path is None:
                rollouts_path = self._root
            assert rollouts_path is not None
            assert rollouts_batch_size is not None
            assert num_rollouts is not None
            self._rollouts_path = os.path.expanduser(
                        os.path.expandvars(rollouts_path))
            self._num_rollouts = num_rollouts
            self._rollouts_batch_size = rollouts_batch_size
            self._load_rollouts()

        # Load attributes
        self._load_attributes()
        in_shape = self._attributes_dict['observations_shape']
        action_units = self._attributes_dict['actions_shape']


        data = {}
        data['in_shape'] = in_shape
        data['action_units'] = action_units
        if 'n_actions' in self._attributes_dict:
            data['n_actions'] = self._attributes_dict['n_actions']
        self._data = data

    def __len__(self) -> int:
        """Returns the number of training samples.
        """
        return self._num_train


    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Accesses the :math:`i^{th}` sample (transition sequence) in the 
        dataset. This consists of 
        the observations, 
        their description (state, or image) if any and
        the transition signals (actions).
        :param idx: index
        :type idx: int
        :return: a numpy array of observations and
                 a numpy array of descrtions (or empty) 
                 a numpy array of transition signals.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        return self._obs[idx], np.array([]), self._actions[idx]


    @property
    def in_shape(self):
        return self._data["in_shape"]


    @property
    def action_units(self) -> int:
        return self._data["action_units"]
    

    @property
    def n_actions(self) -> int:
        """Number of possible actions if discrete. 
        Raises an error if not defined in the attributes. 
        """
        if 'n_actions' not in self._data:
            raise ValueError("n_actions not defined for this dataset")
        return self._data["n_actions"]


    def get_val_batch(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of evaluation samples.

        :return: a tuple of a batch of observation evaluation samples and 
                 transition evaluation samples. 
        :rtype: Tuple[np.ndarray, np.ndarray]
        """    #     imgs = self._images[self.num_train:self.num_train+self.num_val]
    #     transitions = self._transitions[self.num_train:self.num_train+self.num_val]
        return self._val_obs, None, self._val_actions


    def get_example_actions(self) -> Tuple[np.ndarray, np.ndarray]:
        """returns a set of example actions (transition signals) with labels. 

        :return: a tuple of a batch of action signals as perceived by the agent 
                 and associated labels.
        :rtype: Tuple[np.ndarray, np.ndarray]
        """
        # No generic way to do this.
        # Behavior can be refined by an inheriting class.
        raise NotImplementedError("No generic way to get example actions." + 
                                  "This behavior can be defined by the " +
                                  "developer through inheritance.")


    def get_rollouts(self) -> Generator[
         Tuple[npt.NDArray, npt.NDArray], None, None]:
        """Returns a generator of rollouts."""
        
        if not self._rollouts:
            raise ValueError("Rollouts were not loaded.")
        
        b = self._rollouts_batch_size
        for i in range(0, self._roll_obs.shape[0], b): 
            yield self._roll_obs[i:i+b],\
                    self._roll_actions[i:i+b] # type: ignore
            
    def get_n_rollouts(self, n: int) -> Tuple[npt.NDArray, npt.NDArray]:
        """Returns the first n rollouts."""

        if not self._rollouts:
            raise ValueError("Rollouts were not loaded.")
        return self._roll_obs[:n], \
                self._roll_actions[:n] # type: ignore
    

    def _load_data(self) -> None:
        """Loads the data from the hdf5 file.
        """
        with h5py.File(self._root,'r') as f:
            nt = self._num_train
            nv = self._num_val
            n = f[TRANSITIONS][OBSERVATIONS].shape[0]
            if  n < (nt+nv):
                raise ValueError(f"Not enough samples {n} for chosen " + 
                    f"--num_train={nt} and --num_val={nv}")

            self._obs = f[TRANSITIONS][OBSERVATIONS][:self._num_train,:self._n_transitions+1]
            self._actions = f[TRANSITIONS][ACTIONS][:self._num_train,:self._n_transitions] 
            if self._normalize_actions:
                self._M = np.abs(self._actions).max(axis=(0,1))
                self._actions /= self._M
            
            self._val_obs = f[TRANSITIONS][OBSERVATIONS][-nv:,:self._n_transitions+1]
            self._val_actions = f[TRANSITIONS][OBSERVATIONS][-nv:,:self._n_transitions]
            if self._normalize_actions:
                self._val_actions /= self._M


    def _load_attributes(self) -> None:
        """Loads the attributes of the dataset.
        """
        with h5py.File(self._root,'r') as f:
            self._attributes_dict = dict(f.attrs)


    def _load_rollouts(self) -> None:
        """Loads the rollouts data from the hdf5 file.
        """
        with h5py.File(self._rollouts_path,'r') as f:
            self._roll_obs = f[ROLLOUTS][OBSERVATIONS][:self._num_rollouts]
            self._roll_actions = f[ROLLOUTS][ACTIONS][:self._num_rollouts] 
            if self._normalize_actions:
                self._roll_actions /= self._M


    def plot_n_step_reconstruction(self, nets, config, 
                                   device, logger, epoch, figdir)->None:
        """
        Plots the first few transitions in the evaluation batch.

        This method saves the figure in the `figname` path,
        and logs it to WandB as well.
        """

        figname = f'reconstructions.pdf'

        imgs, _, actions = self.get_val_batch()

        plt_utils.plot_n_step_reconstruction(
                imgs, actions, nets, device, logger, 
                plot_on_black=config.plot_on_black, 
                n_steps=self._n_transitions, n_examples=7, 
                savefig=config.savefig, savedir=figdir, 
                log_wandb=config.log_wandb, epoch=epoch, figname=figname)



    def plot_rollout_reconstruction(self, nets, config, device, logger, epoch, 
                                    figdir) -> None:
        """
        Plots the reconstructions of the first :math:`n` rollouts.

        This method saves the figure in the `figname` path,
        and logs it to WandB as well.
        """
        figname = f'rollouts_reconstructions.pdf'

        X, a = self.get_n_rollouts(config.plot_n_rollouts)

        plt_utils.plot_rollout_reconstructions(
                X, a, nets, device, logger, n_rollouts=config.plot_n_rollouts, 
                powers=True, savefig=config.savefig, savedir=figdir, 
                log_wandb=config.log_wandb, epoch=epoch, figname=figname)



if __name__ == '__main__':
    pass