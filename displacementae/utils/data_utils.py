#!/usr/bin/env python3
# Copyright 2022 Hamza Keurti
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
# @title           :data_utils.py
# @author          :Hamza Keurti
# @contact         :hkeurti@ethz.ch
# @created         :08/02/2022
# @version         :1.0
# @python_version  :3.6.6
"""
Miscellaneous data utilities
-----------------------------

Collection of methods used to adjust data to different approaches.
"""

import numpy as np
import torch

def action_to_id(a):
    """
    Converts an action vector in the form of a displacement to an action id.

    Args:
        a (ndarray): a 2D array of actions with type `int` in the form of a 
                        batch of displacement of properties 
                        (\generating factors).
                        Expects values to be in :math:`-1,0,1` with only one 
                        active.
    
    Returns:
        ndarray: a 1D array of type `int`.

    TODO:
        implement in more general cases where action is not necessarily sparse.
    """
    dim = a.shape[-1]
    bases = 1+torch.arange(dim).type_as(a)
    return (torch.relu(a)@(bases) + torch.relu(-a)@(bases+dim)).int() # TODO 

    

