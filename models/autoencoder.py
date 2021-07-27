import torch
import torch.nn as nn
import numpy as np

# import os
# import sys
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models import nets
from models.orthogonal import OrthogonalMatrix


class VariationalOrthogonalAE(nn.Module):
    def __init__(self, img_shape, n_latent, kernel_sizes, strides, conv_channels, hidden_units, intervene=True, rotation_steps=None, device='cpu'):
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
        if self.intervene:
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
                raise Exception(
                    'Expected intervention but no displacement was provided')
            h = self.rotate(h, dz)
        out = self.decode(h)
        return torch.sigmoid(out), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std


class VariationalTranslationAE(nn.Module):
    def __init__(self, img_shape, n_latent, kernel_sizes, strides, conv_channels, hidden_units, intervene=True, device='cpu'):
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

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def translate(self, h, dz):
        return h + dz

    def forward(self, x, dz=None):
        z = self.encode(x)
        mu, logvar = z[:, :self.n_units], z[:, self.n_units:]
        h = self.reparametrize(mu, logvar)
        if self.intervene:
            if dz is None:
                raise Exception(
                    'Expected intervention but no displacement was provided')
            h = self.translate(h, dz)
        out = self.decode(h)
        return torch.sigmoid(out), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std


class VariationalMixAE(nn.Module):
    def __init__(self, img_shape, n_latent, kernel_sizes, strides, conv_channels, hidden_units, intervene=True,rotation_idx = [],translation_idx=[],rotation_steps = None, device='cpu'):
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
        
        self.dz_rotidx = rotation_idx
        self.dz_transidx = translation_idx
        # dz_rotidx are indices to access angle displacement in given dz
        # rotation_units are indices to access latent vectors
        if self.dz_transidx:
            self.translation_units = np.arange(len(self.dz_transidx))

        if self.dz_rotidx:
            self.rotation_units = len(self.dz_transidx) + np.arange(2*len(self.dz_rotidx))
            self.orthogonal = OrthogonalMatrix(
                OrthogonalMatrix.BLOCKS, n_units=len(self.rotation_units), device=device)
        # self.steps = dataset.joint_steps[free_joints].to(device)
        if rotation_steps:
            self.steps = torch.tensor(rotation_steps).to(device)


    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def translate(self, h, dz):
        return h + dz
    
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
                raise Exception(
                    'Expected intervention but no displacement was provided')
            if self.dz_rotidx:
                h[...,self.rotation_units] = self.rotate(h[...,self.rotation_units], dz[...,self.dz_rotidx])
            if self.dz_transidx:
                h[...,self.translation_units] = self.translate(h[...,self.translation_units],dz[...,self.dz_transidx])
        out = self.decode(h)
        return torch.sigmoid(out), mu, logvar

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(mu)
        return mu + eps*std