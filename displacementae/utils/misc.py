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
# @title          :displacementae/utils/misc.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :18/11/2021
# @version        :1.0
# @python_version :3.7.4

from typing import List, Union
import numpy as np

def str_to_ints(str_arg:str) -> List:
    """
    Parse a string argument into a list of ints.
    """
    assert isinstance(str_arg,str)
    str_arg = str_arg.strip()
    str_arg = str_arg.replace(" ","")
    if str_arg in ['','[]'] :
        return []
    # Nested list if ']' not in last position.
    args_list = str_arg.split("],[")
    ret = []
    for args in args_list:
        args = args.replace('[','').replace(']','').replace('"','').split(",")
        args = [int(a.strip()) for a in args]
        ret.append(args)
    
    if len(ret) == 1:
        ret = ret[0]
    return ret

def ints_to_str(args:Union[List[int],int]) -> str:
    assert isinstance(args,list) or isinstance(args, int)
    if isinstance(args, int):
        return str(args)
    else:
        str_ret = ''
        str_ret += ','.join([str(arg) for arg in args])
        return str_ret

def str_to_floats(str_arg:str) -> List:
    """
    Parse a string argument into a list of ints.
    """
    assert isinstance(str_arg,str)
    if str_arg == '':
        return []
    else:
        args = str_arg.replace('[','').replace(']','').replace('"','').split(",")
        args = [float(a.strip()) for a in args]
        return args

def rotation_matrix(yaw,pitch,roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([
        [cy*cp  , cy*sp*sr - sy*cr  , cy*sp*cr + sy*sr],
        [sy*cp  , sy*sp*sr + cy*cr  , sy*sp*cr - cy*sr],
        [-sp    , cp*sr             , cp*cr           ],
    ])
    R = np.moveaxis(R,[0,1],[-2,-1])
    return R