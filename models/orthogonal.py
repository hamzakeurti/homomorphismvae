import torch
import torch.nn as nn
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import nets

def _block_diag(m):
    """
    Taken from `here <https://gist.github.com/yulkang/2e4fc3061b45403f455d7f4c316ab168>`__
    Make a block diagonal matrix along dim=-3
    EXAMPLE:
    block_diag(torch.ones(4,3,2))
    should give a 12 x 8 matrix with blocks of 3 x 2 ones.
    Prepend batch dimensions if needed.
    You can also give a list of matrices.
    :type m: torch.Tensor, list
    :rtype: torch.Tensor
    """
    if type(m) is list:
        m = torch.cat([m1.unsqueeze(-3) for m1 in m], -3)
    d = m.dim()
    n = m.shape[-3]
    siz0 = m.shape[:-3]
    siz1 = m.shape[-2:]
    m2 = m.unsqueeze(-2)
    device = m.device
    eye = _attach_dim(torch.eye(n).unsqueeze(-2), d - 3, 1).to(device)
    return (m2 * eye).reshape(
        siz0 + torch.Size(torch.tensor(siz1) * n)
    )


def _attach_dim(v, n_dim_to_prepend=0, n_dim_to_append=0):
    return v.reshape(
        torch.Size([1] * n_dim_to_prepend)
        + v.shape
        + torch.Size([1] * n_dim_to_append))


class OrthogonalMatrix(nn.Module):
    """
    A torch module that parametrizes a O(n) subgroup, to allow for matrix optimization within the orthogonal group.

    Torch module that parametrizes a group of orthogonal matrices. 
    The module does not have any weights, it just takes a vector of parameters and turns them into an orthogonal matrix.
    """
    CAYLEY = 'cayley'
    BLOCKS = 'blocks'

    def __init__(self, transform=CAYLEY, n_units=6,device = 'cpu'):
        nn.Module.__init__(self)
        self.device = device
        self.transform = transform
        self.n_units = n_units        
        self.matrix_size = n_units
        if transform == OrthogonalMatrix.CAYLEY:
            self.n_parameters = self.n_units*(self.n_units-1)/2
        if transform == OrthogonalMatrix.BLOCKS:
            if self.n_units % 2 == 1:
                raise ValueError(
                    'Latent space should have an even dimension for the matrix to be expressed in blocks of 2.')
            self.n_parameters = self.n_units//2  # Number of blocks
            self.rot_basis = torch.DoubleTensor([
                [[1., 0.],
                 [0., 1.]],
                [[0., -1.],
                 [1., 0.]]]).to(device)

    def forward(self, parameters):
        if parameters.shape[-1] != self.n_parameters:
            raise ValueError(
                f'Expected input last dimension to be {self.n_parameters}, received {parameters.shape[-1]}')
        if self.transform == OrthogonalMatrix.BLOCKS:
            # parameters shape: [b,n_parameters]
            return _block_diag(self.rotation_matrices(parameters))

    def rotation_matrices(self, angle):
        u = torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)
        return torch.tensordot(u, self.rot_basis, dims=[[-1], [0]])


class VariationalOrthogonalAE(nn.Module):
    def __init__(self, img_shape, n_latent, kernel_sizes, strides, conv_channels, hidden_units, rotation_steps=None, device='cpu'):
        super().__init__()
        shape = img_shape
        self.n_units = n_latent
        n_units = n_latent
        self.activate_latent = None
        self.encoder = nets.CNN(shape_in=shape, kernel_sizes=kernel_sizes, strides=strides,
                                conv_channels=conv_channels, linear_channels=hidden_units+[2*n_units], use_bias=True)
        self.decoder = nets.TransposedCNN(shape_out=shape, kernel_sizes=kernel_sizes, strides=strides,
                                          conv_channels=conv_channels[::-1], linear_channels=[n_units] + hidden_units[::-1])
        self.orthogonal = OrthogonalMatrix(
            OrthogonalMatrix.BLOCKS, n_units=n_units, device=device)
        # self.steps = dataset.joint_steps[free_joints].to(device)
        if rotation_steps:
            self.steps = torch.tensor(rotation_steps).to(device)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def rotate(self, h, dz):
        angles = dz*self.steps
        O = self.orthogonal(angles)
        return torch.matmul(O, h.unsqueeze(-1)).squeeze(dim=-1)

    def forward(self, x, dz):
        z = self.encode(x)
        mu, logvar = z[:, :self.n_units], z[:, self.n_units:]
        h = self.reparametrize(mu, logvar)
        z2 = self.rotate(h, dz)
        out = self.decode(z2)
        return torch.sigmoid(out), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std
