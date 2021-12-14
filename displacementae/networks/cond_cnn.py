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
# @title          :displacementae/networks/cond_cnn.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :17/11/2021
# @version        :1.0
# @python_version :3.7.4

import torch
import torch.nn as nn
import networks.cnn as cnn


class ConditionalEncoder(nn.Module):
    """
    Convolutional network that encodes an image conditionally to an embedding vector.
    """
    def __init__(self, n_cond, conv_channels, image_shape, kernel_sizes = 5, strides = 1, out_size = 10, linear_bias = True, device = 'cpu'):
        super().__init__()
        self.device = device
        self.n_cond = n_cond
        self.out_size = out_size
        shape_in = list(image_shape)
        shape_in[0] += n_cond
        conv_channels = [shape_in[0]] + conv_channels
        self.cnn = cnn.CNN(shape_in = shape_in[1:], kernel_sizes = kernel_sizes, strides = strides,conv_channels = conv_channels,linear_channels = None)
        self.linear = nn.Linear(self.cnn._n_conv_out, self.out_size ,bias=linear_bias)


    def forward(self,x,v):
        # Broadcast v to be to be of shape [n_b,n_cond,in_h,in_w]
        n_b,n_cond = v.shape
        _,__,in_h,in_w = x.shape
        v = v.reshape(n_b,n_cond,1,1)
        v = v * torch.ones(1,1,in_h,in_w,dtype=torch.float64).to(self.device)
        
        # Concatenate v and x
        vx =  torch.cat([v,x],dim=1)

        # Forward
        out = vx
        out = self.cnn(out)
        out = out.reshape(n_b, -1)
        out = self.linear(out)
        return out 



