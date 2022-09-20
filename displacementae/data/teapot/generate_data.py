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
# @title          :displacementae/data/teapot/generate_data.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :12/11/2021
# @version        :1.0
# @python_version :3.7.4

import os
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
import h5py


import __init__
import data.teapot.gen_args as gargs
import utils.misc as misc

def read_obj(filename, center=True):
    # Code taken from https://yuchen52.medium.com/beyond-data-scientist-3d-plots-in-python-with-examples-2a8bd7aa654b
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f": # face data
                # e.g. "f 1/1/1/ 2/2/2 3/3/3 4/4/4 ..."
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v": # vertex data
                # e.g. "v  30.2180 89.5757 -76.8089"
                vertex = list(map(lambda c: float(c), components[1:]))
                vertices.append(vertex)
    vertices = np.array(vertices)
    if center:
        vertices -= vertices.mean(axis=0)[None,:]
    return vertices, np.array(triangles)



def rotation_matrix(yaw,pitch,roll):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    R = np.array([
        [cy*cp  , cy*sp*sr - sy*cr  , cy*sp*cr + sy*sr],
        [sy*cp  , sy*sp*sr + cy*cr  , sy*sp*cr - cy*sr],
        [-sp    , cp*sr             , cp*cr           ],
    ])
    return R



def sample_orientations_from_canonical(
                    vertices, batch_size, mode='continuous',
                    n_values=None):
    """
    Produce a batch of different orientations from a single starting orientation.
    """
    if mode == 'discrete':
        p = np.random.randint(n_values,size=[batch_size,3],dtype=float)/(n_values-1)  
    elif mode == 'continuous':
        p = np.random.random([batch_size,3])
    p = 2*np.pi*p
    R = rotation_matrix(*p.T)
    v_out = np.einsum('ijb,vj->bvi',R,vertices)
    return v_out,p



def sample_orientations_from_orientations(vertices, mode='continuous',
                                          n_values=None, low=-np.pi, 
                                          high=np.pi):
    """
    Produce a batch of orientations by rotating a batch of orientations.
    """
    batch_size = vertices.shape[0]
    if mode == 'discrete':
        p = np.random.randint(n_values,size=[batch_size,3],dtype=float)/(n_values-1)  
    elif mode == 'continuous':
        p = np.random.random([batch_size,3])
    range = high-low
    p *= range
    p += low

    R = rotation_matrix(*p.T)
    v_out = np.einsum('ijb,bvj->bvi',R,vertices)
    return v_out,p



def sample_n_steps_orientations_from_canonical(
            vertices, batch_size, n_steps=2, mode='continuous',n_values=None,
            action_range=(-np.pi,np.pi)):
    
    low, high = action_range
    v_out = np.zeros(
        shape=(batch_size, n_steps+1, *vertices.shape),
        dtype=np.half)
    
    # First action angles corresponds to rotation from the canonical view of 
    # the first image
    a_out = np.zeros(
        shape=(batch_size, n_steps+1, 3),
        dtype=np.half)
    # Sample initial positions
    v_out[:,0,...], a_out[:,0,...] = sample_orientations_from_canonical(
                    vertices, batch_size, mode=mode,
                    n_values=n_values)
    # for step
    #    sample orientations
    for step in range(n_steps):
        v_out[:,step+1,...], a_out[:,step+1,...] = sample_orientations_from_orientations(
                    vertices=v_out[:,step,...],
                    mode=mode, n_values=n_values, low=low, high=high)


    return v_out, a_out

def get_image(vertices, triangles, figsize=(3,3), dpi=36):
    """
    Plots a 3D view of the object and returns it as a numpy array.
    """
    fig = plt.figure(figsize=figsize,dpi=dpi)   
    ax = plt.axes(projection='3d')

    lim = 1.5
    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_zlim([-lim,lim])

    ax.set_axis_off()

    x,y,z = vertices[:,0], vertices[:,1], vertices[:,2]
    ax.plot_trisurf(x,z,triangles,y,shade=True,color='white') 

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(figsize[0]*dpi,figsize[1]*dpi,3)
    # image /= 255
    # image = np.moveaxis(image,-1,0)
    plt.close()
    return image


def vertices_to_images(v, triangles, figsize=(3,3), dpi=24):
    """
    Converts a batch of vertices arrays to a batch of figures 

    Args:
        v (ndarray): a 2D array of actions with type `int` in the form of a 
                        batch of displacement of properties 
                        (\generating factors).
                        Expects values to be in :math:`-1,0,1` with only one 
                        active.
        triangles (ndarray): Describes the faces of the solid. 
                        The same triangles array is used for all 
                        transformed versions of the solid.
        figsize (tuple): in inches, describes the figure size for each plot.
        dpi (int):      dots per inch, number of pixels per inch in the figure, 

    Returns:
        ndarray: a batch of images.

    """
    # v is of shape [batch, n_steps, n_vertices, 3]
    h,w = figsize[0]*dpi,  figsize[1]*dpi
    images_out = np.zeros(shape=[*v.shape[:-2],h,w,3])
    # v = v.reshape([-1,*v.shape[-2:]])
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            images_out[i,j] = get_image(v[i,j], triangles,
                                        figsize=figsize, dpi=dpi)
    images_out /= 255
    images_out = np.moveaxis(images_out,-1,-3)
    return images_out



def generate_dataset(obj_filename, out_path, batch_size, figsize=(3,3), dpi=24, 
                     mode='continuous', n_values=None, 
                     action_range=[-np.pi/2,np.pi/2],
                     n_steps=2, n_samples=10000):
    vertices, triangles = read_obj(obj_filename)
    with h5py.File(out_path, "w") as f:
        dset_img = f.create_dataset('images', 
                shape=(n_samples, n_steps+1, 3, figsize[0]*dpi, figsize[1]*dpi), 
                maxshape=(None, n_steps+1, 3, figsize[0]*dpi, figsize[1]*dpi),
                dtype=np.float32)
        dset_rot = f.create_dataset('rotations', 
                shape=(n_samples, n_steps+1, 3), 
                maxshape=(None, n_steps+1, 3),
                dtype=np.float32)

        n_batches = n_samples//batch_size
        for i in range(n_batches):
            print(f'Sampling batch {i}/{n_batches}')
            v, a = sample_n_steps_orientations_from_canonical(
                    vertices, batch_size=batch_size,
                    n_steps=n_steps, mode=mode,
                    n_values=n_values, action_range=action_range)
            images = vertices_to_images(
                    v, triangles, figsize=figsize, dpi=dpi)
            dset_img[i*batch_size:(i+1)*batch_size] = images
            dset_rot[i*batch_size:(i+1)*batch_size] = a
    return 


if __name__=='__main__':

    config = gargs.parse_cmd_arguments()
    if not os.path.exists(os.path.dirname(config.out_path)):
        os.makedirs(os.path.dirname(config.out_path))
    figsize = misc.str_to_ints(config.figsize)
    action_range = misc.str_to_floats(config.action_range)
    if config.mode == 'continuous':
        config.n_values = None
    
    generate_dataset(obj_filename=config.obj_filename, 
                     out_path=config.out_path, 
                     batch_size=config.batch_size, 
                     figsize=figsize, 
                     dpi=config.dpi, 
                     mode=config.mode, 
                     n_values=config.n_values, 
                     action_range=action_range,
                     n_steps=config.n_steps, 
                     n_samples=config.n_samples)