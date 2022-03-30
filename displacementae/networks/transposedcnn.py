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
# @title          :displacementae/networks/transposedcnn.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransposedCNN(nn.Module):
    def __init__(self, shape_out, kernel_sizes=5, strides=1, conv_channels=[64, 64, 32, 1], linear_channels=None, use_bias=True, activation_fn=torch.relu):
        """
        Transposed CNN, linear readout followed by transposed convolutions.
        Serves as an expanding path.

        Args:
            shape_in (tuple, optional): [description]. Defaults to (64, 64).
            kernel_sizes (int, optional): [description]. Defaults to 5.
            conv_channels (list, optional): [description]. Defaults to [32, 64, 64].
            linear_channels (list, optional): [description]. Defaults to [20].
            use_bias (bool, optional): [description]. Defaults to True.
        """
        nn.Module.__init__(self)
        self._activation_fn = activation_fn
        self._use_bias = use_bias

        self._shape_out = shape_out
        self._conv_channels = conv_channels
        n_conv = len(self._conv_channels)-1

        self._kernel_sizes = kernel_sizes
        if not isinstance(kernel_sizes, list):
            self._kernel_sizes = [kernel_sizes for i in range(n_conv)]

        self._strides = strides
        if not isinstance(strides, list):
            self._strides = [strides for i in range(n_conv)]

        self._conv_layers = nn.ModuleList()
        for l in range(n_conv):
            self._conv_layers.append(nn.ConvTranspose2d(
                in_channels=self._conv_channels[l],
                out_channels=self._conv_channels[l+1],
                kernel_size=self._kernel_sizes[l],
                stride= self._strides[l],
                bias=self._use_bias))


        self._fm_shapes = [list(shape_out)]
        for l in range(n_conv-1, -1, -1):
            h, w = self._fm_shapes[0]
            new_h = (h - self._kernel_sizes[l])//self._strides[l] + 1
            new_w = (w - self._kernel_sizes[l])//self._strides[l] + 1
            self._fm_shapes = [[new_h, new_w]] + self._fm_shapes

        n_lin = 0 if linear_channels is None else len(linear_channels)
        self._lin_layers = nn.ModuleList()
        self._n_conv_in = np.prod(self._fm_shapes[0],dtype=int)*self._conv_channels[0]
        if linear_channels is not None:
            self._lin_channels = linear_channels + [self._n_conv_in]
            for l in range(n_lin):
                self._lin_layers.append(nn.Linear(
                    self._lin_channels[l],
                    self._lin_channels[l+1],
                    self._use_bias))

    def forward(self, x):
        out = x
        # Linear layers
        for l in range(len(self._lin_layers)):
            out = self._lin_layers[l](out)
            if self._activation_fn is not None:
                out = self._activation_fn(out)

        if len(self._lin_layers) != 0:
            out = out.view([-1, self._conv_channels[0]]+self._fm_shapes[0])

        # Transposed Convolutions
        for l in range(len(self._conv_layers)):
            out = self._conv_layers[l](out)
            is_last_layer = l == (len(self._conv_layers) - 1)
            if (not is_last_layer) and (self._activation_fn is not None):
                out = self._activation_fn(out)

        return out