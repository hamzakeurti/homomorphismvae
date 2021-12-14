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
# @title          :displacementae/networks/cond_transposedcnn.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :14/12/2021
# @version        :1.0
# @python_version :3.7.4


import torch
import torch.nn as nn
import networks.transposedcnn as transposedcnn


class ConditionalDecoder(nn.Module):
    """
    Transposed Convolutional network that decodes conditionally to an embedding vector.
    """
    def __init__(self,in_size, n_cond, conv_channels, image_shape, kernel_sizes = 5, strides = 1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.n_cond = n_cond
        self.in_size = in_size 
        self.conv_channels = conv_channels + [image_shape[0]]
        self.transcnn = transposedcnn.TransposedCNN(shape_out=image_shape[1:], kernel_sizes=kernel_sizes, strides=strides, conv_channels=self.conv_channels)
        self.linear = nn.Linear(in_size + n_cond, self.transcnn._n_conv_in)


    def forward(self,z,v):

        # Concatenate z and v
        zv =  torch.cat([z,v],dim=-1)

        # Forward
        out = zv
        out = self.linear(out)
        linout_shape = (-1,self.transcnn._conv_channels[0],*self.transcnn._fm_shapes[0])
        out = out.reshape(linout_shape)
        out = self.transcnn(out)
        return out 
