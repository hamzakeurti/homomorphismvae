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
import numpy.typing as npt
import mujoco


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


def euler_to_mat(euler: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert euler angles ZYX (roll, pitch, yaw) to rotation matrix.

    Args:
        euler (npt.NDArray[np.float64]): euler angles ZYX (..., 3)
    
    Returns:
        npt.NDArray[np.float64]: rotation matrix (..., 9)
    """
    cr = np.cos(euler[..., 0])
    sr = np.sin(euler[..., 0])
    cp = np.cos(euler[..., 1])
    sp = np.sin(euler[..., 1])
    cy = np.cos(euler[..., 2])
    sy = np.sin(euler[..., 2])

    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    R = np.moveaxis(R, [0, 1], [-2, -1])
    return R.reshape(euler.shape[:-1] + (9,))


# convert euler angles ZYX numpy array (possibly a batch) to quaternions
# zyx euler angles are yaw, pitch, roll
def euler_to_quat(euler: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    cr = np.cos(euler[..., 0] * 0.5)
    sr = np.sin(euler[..., 0] * 0.5)
    cp = np.cos(euler[..., 1] * 0.5)
    sp = np.sin(euler[..., 1] * 0.5)
    cy = np.cos(euler[..., 2] * 0.5)
    sy = np.sin(euler[..., 2] * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return np.stack([qw, qx, qy, qz], axis=-1)

# convert quaternions numpy array (possibly a batch) to euler angles ZYX
def quat_to_euler(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion to euler angles (roll, pitch, yaw)
    
    Args:
        quat (npt.NDArray[np.float64]): quaternion array shape (..., 4)

    Returns:
        npt.NDArray[np.float64]: euler angles roll, pitch, yaw (..., 3)
    """
    
    qw, qx, qy, qz = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = np.where(np.abs(sinp) >= 1, np.sign(sinp) * np.pi / 2, np.arcsin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.stack([roll, pitch, yaw], axis=-1)


def quat_to_mat(quat: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert quaternion to rotation matrix using mujoco

    Args:
        quat (npt.NDArray[np.float64]): quaternion array shape (..., 4)

    Returns:
        npt.NDArray[np.float64]: rotation matrix (..., 9)
    """
    q = quat.reshape(-1, 4)
    ret = np.zeros(q.shape[:-1] + (9,))
    for i in range(q.shape[0]):
        mujoco.mju_quat2Mat(ret[i],q[i])
    return ret.reshape(quat.shape[:-1] + (9,))


def hue_to_rgb(hue: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Convert hue to rgb

    Args:
        hue (npt.NDArray[np.float64]): hue array shape (..., 1)

    Returns:
        npt.NDArray[np.float64]: rgb array shape (..., 3)
    """
    ph = 2*np.pi/3
    rgb = np.stack([np.cos(2*np.pi*hue),
                       np.cos(2*np.pi*hue+ph),
                       np.cos(2*np.pi*hue+2*ph)], axis=-1)
    return rgb