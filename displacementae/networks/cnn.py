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
# @title          :displacementae/networks/cnn.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CNN(nn.Module):

    def __init__(self, shape_in, kernel_sizes=5, strides=1, 
                conv_channels=[1, 32, 64, 64], linear_channels=None, 
                use_bias=True, activation_fn=torch.relu):
        """
        A convolutional neural network with convolutional layers 
        followed by a linear readout.

        Args:
            shape_in (tuple, optional): tuple of height, width of the input, 
                channels should be specified in conv_channels argument.
            kernel_sizes (int, optional): [description]. Defaults to 5.
            conv_channels (list, optional): [description].
            linear_channels (list, optional): [description]. Defaults to [20].
            use_bias (bool, optional): [description]. Defaults to True.
            use_bn (bool, optional): [description]. Defaults to True.
        """
        nn.Module.__init__(self)
        self._activation_fn = activation_fn
        self._use_bias = use_bias
 
        self._shape_in = shape_in
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
            self._conv_layers.append(nn.Conv2d(
                in_channels=self._conv_channels[l],
                out_channels=self._conv_channels[l+1],
                kernel_size=self._kernel_sizes[l],
                stride=self._strides[l],
                bias=self._use_bias))

        self._fm_shapes = [list(shape_in)]
        for l in range(n_conv):
            h, w = self._fm_shapes[l]
            new_h = (h - self._kernel_sizes[l])//self._strides[l] + 1
            new_w = (w - self._kernel_sizes[l])//self._strides[l] + 1
            self._fm_shapes.append([new_h, new_w])

        n_lin = 0 if linear_channels is None else len(linear_channels)
        self._lin_layers = nn.ModuleList()
        self._n_conv_out = np.prod(self._fm_shapes[-1],dtype=int)*self._conv_channels[-1]
        if linear_channels is not None:
            self._lin_channels = [self._n_conv_out] + linear_channels
            for l in range(n_lin):
                self._lin_layers.append(nn.Linear(
                    self._lin_channels[l],
                    self._lin_channels[l+1],
                    self._use_bias))

    def forward(self, x):
        out = x
        # Convolutions
        for l in range(len(self._conv_layers)):
            out = self._conv_layers[l](out)
            is_last_layer = (l == len(self._conv_layers) -
                             1) and (len(self._lin_layers) == 0)
            if (not is_last_layer) and (self._activation_fn is not None):
                out = self._activation_fn(out)

        # Linear layers
        if len(self._lin_layers) != 0:
            out = out.view([-1, self._lin_channels[0]])
        for l in range(len(self._lin_layers)):
            out = self._lin_layers[l](out)
            is_last_layer = (l == len(self._lin_layers) - 1)
            if (not is_last_layer) and (self._activation_fn is not None):
                out = self._activation_fn(out)

        return out

