from typing import List
import torch
import torch.nn as nn

import numpy as np

from grouprepr.group_representation import GroupRepresentation
from networks.mlp import MLP


class UnstructuredRepresentation(GroupRepresentation):
    """
    Class for unstructured latent transformation.

    Instead of transforming latent by group actions, this representation
    simply map [latent, action] to the next latent through an MLP
    """
    def __init__(self,
                 n_action_units: int,
                 dim_representation: int,
                 hidden_units: List[int] = [50, 50],
                 activation=nn.ReLU(),
                 layer_norm=False,
                 device='cpu',
                 varphi_units:list=[]) -> None:

        super().__init__(n_action_units, dim_representation, device=device,
                         varphi_units=varphi_units)

        self.net = MLP(in_features=dim_representation + self.varphi_out,
                       out_features=dim_representation,
                       hidden_units=hidden_units,
                       activation=activation,
                       dropout_rate=0,
                       bias=True,
                       layer_norm=layer_norm).to(device)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        """
        Gets the representation matrix of input transition :arg:`a`.
        """
        pass

    def act(self, a: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Acts on an input representation vector :arg:`z` through matrix
        product with the representation matrix of input transition
        :arg:`a`.

        Args:
            a, torch.Tensor: Batch of transitions.
                        shape: `[batch_size,n_action]`
            z, torch.Tensor: Batch of representation vectors.
                        shape: `[batch_size,n_repr]`
        Returns:
            torch.Tensor: Transformed representation vectors.
                        shape: `[batch_size,n_repr_units]`
        """
        a = self.varphi(a)
        a_embed = nn.functional.one_hot(a.long(), self.n_action_units) if len(a.shape) == 1 else a
        return self.net(torch.cat([z, a_embed], dim=1))

    def get_example_repr(self, a: torch.Tensor = None) -> np.ndarray:
        raise NotImplementedError

    def representation_loss(self, *args):
        pass

    def end_iteration(self):
        pass
