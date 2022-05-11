import numpy as np
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, data_path, data_type, rseed=None):

        super().__init__()
        self._data_path = data_path
        self._data = np.load(data_path)

        if data_type == 'teapot':
            self._imgs = np.moveaxis(self._data['imgs'], 4, 2) / 255.
        elif data_type == 'dsprites':
            self._imgs = np.expand_dims(self._data['imgs'], axis=2)
        else:
            raise ValueError
        self._actions = self._data['actions']
        self.n_actions = int(self._data['n_actions'])
        self.in_shape = self._imgs.shape[2:]
        self.action_shape = [1]

        if rseed is not None:
            self._rand = np.random.RandomState(rseed)
        else:
            self._rand = np.random

    def __getitem__(self, idx):
        return self._imgs[idx], self._actions[idx]

    def __len__(self):
        return len(self._imgs)

    def get_example_actions(self):
        return 0, 0
