import torch
import torch.nn as nn

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import nets
from models.orthogonal import OrthogonalMatrix


class VariationalOrthogonalAE(nn.Module):
    def __init__(self, img_shape,n_latent, kernel_sizes, strides, conv_channels, hidden_units, intervene = True, rotation_steps=None, device='cpu'):
        super().__init__()
        shape = img_shape
        self.n_units = n_latent
        n_units = n_latent
        self.activate_latent = None
        self.intervene = intervene
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

    def forward(self, x, dz=None):
        z = self.encode(x)
        mu, logvar = z[:, :self.n_units], z[:, self.n_units:]
        h = self.reparametrize(mu, logvar)
        if self.intervene:
            if dz is None:
                raise Exception('Expected intervention but no displacement was provided') 
            h = self.rotate(h, dz)
        out = self.decode(h)
        return torch.sigmoid(out), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std
