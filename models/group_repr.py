

"""
Group representations
---------------------
Mappings of group actions of the environment in the latent space.
Mappings onto different group structures.
Mappings can be learnable.
Mappings are ideally group morphisms.
"""

import torch
import torch.nn as nn

from models.orthogonal import OrthogonalMatrix


class AdaptiveReprBlock(nn.Module):
    def __init__(self,n_repr_units,learn_repr=True,repr_init_scale=0.1,device='cpu'):
        nn.Module.__init__(self)
        self.n_repr_units = n_repr_units
        self.learn_repr = learn_repr
        self.repr_init_scale = repr_init_scale
        self.device = device
        self.unit_repr = nn.Parameter(torch.ones(self.n_repr_units,dtype=torch.float,device=self.device)*repr_init_scale,requires_grad=self.learn_repr)
    def forward(self,action):
        pass

class AdaptiveRotationBlock(AdaptiveReprBlock):
    """
    Block that learns a rotation group representation of input actions.
    Representation of actions are rotation matrices that are in group morphism with the action group.

    The representation of elementary action is learned and the representation of other actions is obtained from composition and inverse.
    The block receives an action scalar and returns a rotation matrix representant of the action.

    Args:

    """
    def __init__(self,n_units,learn_repr=True,repr_init_scale=0.1,device='cpu'):
        AdaptiveReprBlock.__init__(self,n_units//2,learn_repr,repr_init_scale,device)
        if n_units % 2 == 1:
            raise ValueError(
                'Latent space should have an even dimension for the matrix to be expressed in blocks of 2.')
        self.n_angles = n_units//2
        self.orthogonal = OrthogonalMatrix(
                OrthogonalMatrix.BLOCKS, n_units=n_units, device=self.device)
        
    def forward(self,action):
        """Compute the representation of the action as a rotation matrix of the latent space.
        Example:
        case where action is a scalar.
        :math:`a = k*e`
        :math:`\rho(a)=\rho(e)^k`
        :math:`\rho(e)` is internally maintained.
        Input: action, shape: [n_batch,n_actions]
        Output: rot_matrices in block diagonal, shape: [n_batch,2*n_actions,2*n_actions]
        """
        angles = action*self.unit_repr
        return self.orthogonal(angles)

    def rotate(self,h,action):
        O = self.forward(action)
        return torch.matmul(O, h.unsqueeze(-1)).squeeze(dim=-1)

class AdaptiveTranslationBlock(AdaptiveReprBlock):
    """
    Block that learns a translation group representation of input actions.
    Representation of actions are translation vectors that are in group morphism with the action group.

    """
    def __init__(self,n_units,learn_repr=True,repr_init_scale=0.1,device='cpu'):
        AdaptiveReprBlock.__init__(self,n_units,learn_repr,repr_init_scale,device)
    
    def forward(self,action):
        return action * self.unit_repr
    
    def translate(self,h,action):
        return h + self.forward(action)
