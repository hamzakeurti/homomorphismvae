
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    """Simple MLP network

    Network made of stacked linear layers, optionally activated.
    """

    def __init__(self, in_features, out_features, hidden_units, activation=torch.relu, dropout_rate=0, bias=True):
        super().__init__()
        self._layers = nn.ModuleList()

        units = [in_features] + hidden_units + [out_features]
        for l in range(len(units)-1):
            n_in = units[l]
            n_out = units[l + 1]
            self._layers.append(
                nn.Linear(in_features=n_in, out_features=n_out, bias=bias))

        self._activation = activation
        self._dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """Computes the output of this network.

        Args:
            x (tensor): 

        Returns:
            tensor: output tensor, not activated.
        """
        h = x
        for l in range(len(self._layers)):
            h = self._layers[l](h)
            if l != len(self._layers) - 1:
                h = self._dropout(h)
                if self._activation is not None:
                    h = self._activation(h)
        return h


class CNN(nn.Module):

    def __init__(self, shape_in, kernel_sizes=5, strides=1, conv_channels=[1, 32, 64, 64], linear_channels=None, use_bias=True, use_bn=True, activation_fn=torch.relu):
        """[summary]

        Args:
            shape_in (tuple, optional): [description]. Defaults to (64, 64).
            kernel_sizes (int, optional): [description]. Defaults to 5.
            conv_channels (list, optional): [description]. Defaults to [32, 64, 64].
            linear_channels (list, optional): [description]. Defaults to [20].
            use_bias (bool, optional): [description]. Defaults to True.
            use_bn (bool, optional): [description]. Defaults to True.
        """
        nn.Module.__init__(self)
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self._use_bn = use_bn

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

        # TODO: implement bn
        self._bn_layers_2D = nn.ModuleList()
        self._bn_layers_1D = nn.ModuleList()

        self._fm_shapes = [list(shape_in)]
        for l in range(n_conv):
            h, w = self._fm_shapes[l]
            new_h = (h - self._kernel_sizes[l])/self._strides[l] + 1
            new_w = (w - self._kernel_sizes[l])/self._strides[l] + 1
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


class TransposedCNN(nn.Module):
    def __init__(self, shape_out, kernel_sizes=5, strides=1, conv_channels=[64, 64, 32, 1], linear_channels=None, use_bias=True, use_bn=True, activation_fn=torch.relu):
        """[summary]

        Args:
            shape_in (tuple, optional): [description]. Defaults to (64, 64).
            kernel_sizes (int, optional): [description]. Defaults to 5.
            conv_channels (list, optional): [description]. Defaults to [32, 64, 64].
            linear_channels (list, optional): [description]. Defaults to [20].
            use_bias (bool, optional): [description]. Defaults to True.
            use_bn (bool, optional): [description]. Defaults to True.
        """
        nn.Module.__init__(self)
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self._use_bn = use_bn

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
                bias=self._use_bias))

        # TODO: implement bn
        self._bn_layers_2D = nn.ModuleList()
        self._bn_layers_1D = nn.ModuleList()

        self._fm_shapes = [list(shape_out)]
        for l in range(n_conv-1, -1, -1):
            h, w = self._fm_shapes[0]
            new_h = int((h - self._kernel_sizes[l])/self._strides[l] + 1)
            new_w = int((w - self._kernel_sizes[l])/self._strides[l] + 1)
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
        self.cnn = CNN(shape_in = shape_in[1:], kernel_sizes = kernel_sizes, strides = strides,conv_channels = conv_channels,linear_channels = None)
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


class ConditionalDecoder(nn.Module):
    """
    Convolutional network that encodes an image conditionally to an embedding vector.
    """
    def __init__(self,in_size, n_cond, conv_channels, image_shape, kernel_sizes = 5, strides = 1, device = 'cpu'):
        super().__init__()
        self.device = device
        self.n_cond = n_cond
        self.in_size = in_size 
        self.conv_channels = conv_channels + [image_shape[0]]
        self.transcnn = TransposedCNN(shape_out=image_shape[1:], kernel_sizes=kernel_sizes, strides=strides, conv_channels=self.conv_channels)
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

