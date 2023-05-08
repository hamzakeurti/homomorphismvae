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
import matplotlib.colors as clr
import numpy as np
from mpl_toolkits import mplot3d
import h5py
from typing import List

import __init__
import displacementae.data.obj3d.gen_args as gargs
import displacementae.utils.misc as misc

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
    return np.moveaxis(R,-1,0)


def sample_orientations_from_canonical(
                    vertices, batch_size, mode='continuous',
                    n_values=None, rotation_matrix_action:bool=False,
                    rots_range:List=(-np.pi,np.pi)):
    """
    Produce a batch of different orientations from a single starting orientation.
    """
    # if mode == 'discrete':
    #     p = np.random.randint(n_values,size=[batch_size,3],dtype=float)/(n_values-1)  
    # elif mode == 'continuous':
    p = np.random.random([batch_size,3])
    m,M = rots_range
    p = (M-m)*p + m
    R = rotation_matrix(*p.T)
    v_out = np.einsum('bij,vj->bvi',R,vertices)
    if rotation_matrix_action:
        return v_out, R.reshape(-1,9), R.reshape(-1,9)
    else:
        return v_out, p, R.reshape(-1,9)
        


def sample_orientations_from_orientations(vertices, pos_in, mode='continuous',
                                          n_values=None, low=-np.pi, 
                                          high=np.pi,
                                          rotation_matrix_action:bool=False):
    """
    Produce a batch of orientations by rotating a batch of orientations.
    """
    batch_size = vertices.shape[0]
    if mode == 'discrete':
        p = np.random.randint(n_values,size=[batch_size,3])/(n_values-1)  
    elif mode == 'continuous':
        p = np.random.random([batch_size,3])
    range = high-low
    p *= range
    p += low

    R = rotation_matrix(*p.T)
    v_out = np.einsum('bij,bvj->bvi',R,vertices)

    R_in = pos_in[:,:9].reshape(-1,3,3)
    R_out = (R@R_in).reshape(-1,9) 

    if rotation_matrix_action:
        return v_out, R.reshape(-1,9), R_out
    else:
        return v_out, p, R_out


def sample_poses_from_canonical(
            vertices, batch_size, mode='continuous', n_values=None, 
            rots_range=(-np.pi,np.pi), 
            translation_grid=5, translation_stepsize=0.5,
            rotation_matrix_action:bool=False):
    # if mode == 'discrete':
    #     p = np.random.randint(n_values,size=[batch_size,3],dtype=float)/(n_values-1)  
    # elif mode == 'continuous':
    p = np.random.random([batch_size,3])
    m,M = rots_range
    p = (M-m)*p + m
    R = rotation_matrix(*p.T)
    v_out = np.einsum('bij,vj->bvi',R,vertices)
    t = (np.random.randint(translation_grid*2+1, size=[batch_size,3]) - translation_grid)*translation_stepsize
    v_out += t[:,None,:]
    if rotation_matrix_action:
        p_out = np.hstack([R.reshape(-1,9),t])
    else:
        p_out = np.hstack([p,t])
    pos_out = np.hstack([R.reshape(-1,9),t])
    return v_out, p_out, pos_out


def sample_trans_from_canonical(vertices, batch_size, translation_grid=5, translation_stepsize=0.5):
    t = (np.random.randint(translation_grid*2+1, size=[batch_size,3]) - translation_grid)*translation_stepsize
    v_out = vertices[None,:,:] + t[:,None,:]
    return v_out, t, t


def sample_poses_from_poses(vertices, pos_in, mode='continuous',n_values=None, 
                            low=-np.pi, high=np.pi ,translation_grid=5, 
                            translation_stepsize=0.5, 
                            translation_range=1, 
                            rotation_matrix_action:bool=False):
    batch_size = vertices.shape[0]
    if mode == 'discrete':
        p = np.random.randint(n_values,size=[batch_size,3],dtype=float)/(n_values-1)  
    elif mode == 'continuous':
        p = np.random.random([batch_size,3])
    range = high-low
    p *= range
    p += low

    R = rotation_matrix(*p.T)
    
    g = translation_grid
    s =translation_stepsize
    r = translation_range
    t = (np.random.randint(r*2+1, size=[batch_size,3]) - r)* s
    
    # recenter rotate then translate back to new translated pos
    pos = vertices.mean(axis=1)
    v_out = vertices - pos[:,None,:]
    v_out = np.einsum('bij,bvj->bvi',R,v_out) # rotate
    
    # cyclic tranlations
    new_pos = np.round((pos + t)/s + g) % (2*g+1) # FIX
    new_pos = (new_pos - g)*s
    v_out += new_pos[:,None,:]
    
    R_in = pos_in[:,:9].reshape(-1,3,3)
    R_out = (R@R_in).reshape(-1,9) 
    pos_out = np.hstack([R_out, new_pos/translation_stepsize]) 
    if rotation_matrix_action:
        p_out = np.hstack([R.reshape(-1,9),t])
    else:
        p_out = np.hstack([p,t])
    return v_out, p_out, pos_out


def sample_trans_from_trans(vertices, translation_grid=5, translation_stepsize=0.5, translation_range=1):
    batch_size = vertices.shape[0]

    g = translation_grid
    s =translation_stepsize
    r = translation_range
    t = (np.random.randint(r*2+1, size=[batch_size,3]) - r)* s
    
    # recenter then translate back to new translated pos
    pos = vertices.mean(axis=1)
    v_out = vertices - pos[:,None,:]
    
    # cyclic tranlations
    new_pos = np.round((pos + t)/s + g) % (2*g+1) 
    new_pos = (new_pos - g)*s
    v_out += new_pos[:,None,:]
    
    return v_out,t, new_pos/translation_stepsize


def sample_n_steps_orientations_from_canonical(
            vertices, batch_size:int, n_steps:int=2, 
            mode:str='continuous',
            n_values:int=None, 
            rots_range:List=(-np.pi,np.pi), 
            rots_range_canonical:List=(-np.pi,np.pi), 
            rotation_matrix_action:List=False):
    
    low, high = rots_range
    n_actions = 9 if rotation_matrix_action else 3

    v_out = np.zeros(
        shape=(batch_size, n_steps+1, *vertices.shape),
        dtype=np.float32)
    
    # First action angles corresponds to rotation from the canonical view of 
    # the first image
    a_out = np.zeros(
        shape=(batch_size, n_steps+1, n_actions),
        dtype=np.float32)

    pos_out = np.zeros(
        shape=(batch_size, n_steps+1, 9),
        dtype=np.float32)
    
    # Sample initial positions
    v_out[:,0,...], a_out[:,0,...],pos_out[:,0,...] = \
            sample_orientations_from_canonical(
                    vertices, batch_size, mode=mode,
                    n_values=n_values, 
                    rotation_matrix_action=rotation_matrix_action, 
                    rots_range=rots_range_canonical)
    # for step
    #    sample orientations
    for step in range(n_steps):

        v_out[:,step+1,...], a_out[:,step+1,...], pos_out[:,step+1,...] = \
                sample_orientations_from_orientations(
                    vertices=v_out[:,step], pos_in=pos_out[:,step],
                    mode=mode, n_values=n_values, low=low, high=high,
                    rotation_matrix_action=rotation_matrix_action)
    return v_out, a_out, pos_out


def sample_n_steps_poses_from_canonical(
            vertices, batch_size, n_steps=2, mode='continuous', n_values=None,
            rots_range=(-np.pi,np.pi),
            rots_range_canonical=(-np.pi,np.pi), 
            translation_grid=3,
            translation_stepsize=0.5, translation_range=1, 
            rotation_matrix_action:bool=False):
    
    low, high = rots_range
    n_actions = 3 # 3D translation
    n_actions += 9 if rotation_matrix_action else 3

    v_out = np.zeros(
        shape=(batch_size, n_steps+1, *vertices.shape),
        dtype=np.float32)
    
    # First action angles corresponds to rotation from the canonical view of 
    # the first image
    a_out = np.zeros(
        shape=(batch_size, n_steps+1, n_actions),
        dtype=np.float32)
    pos_out = np.zeros(
        shape=(batch_size, n_steps+1, 12), # always returns rotation matrix
        dtype=np.float32)
    # Sample initial positions
    v_out[:,0,...], a_out[:,0,...], pos_out[:,0,...] =\
         sample_poses_from_canonical(
                    vertices, batch_size, mode=mode,
                    n_values=n_values, rots_range=rots_range_canonical,
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,
                    rotation_matrix_action=rotation_matrix_action)
    # for step
    #    sample poses
    for step in range(n_steps):

        v_out[:,step+1,...], a_out[:,step+1,...], pos_out[:,step+1,...] =\
             sample_poses_from_poses(
                    vertices=v_out[:,step], pos_in=pos_out[:,step],
                    mode=mode, n_values=n_values, low=low, high=high,
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,
                    translation_range=translation_range)
    return v_out, a_out, pos_out


def sample_n_steps_trans_from_canonical(
            vertices, batch_size, n_steps=2, translation_grid=3,translation_stepsize=0.5,translation_range=1):
    
    v_out = np.zeros(
        shape=(batch_size, n_steps+1, *vertices.shape),
        dtype=np.float32)
    
    # First action angles corresponds to rotation from the canonical view of 
    # the first image
    a_out = np.zeros(
        shape=(batch_size, n_steps+1, 3),
        dtype=np.float32)
    pos_out = np.zeros(
        shape=(batch_size, n_steps+1, 3),
        dtype=np.float32)
    # Sample initial positions
    v_out[:,0,...], a_out[:,0,...], pos_out[:,0,...] =\
         sample_trans_from_canonical(
                    vertices, batch_size,
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,)
    # for step
    #    sample poses
    for step in range(n_steps):

        v_out[:,step+1,...], a_out[:,step+1,...], pos_out[:,step+1,...] =\
             sample_trans_from_trans(
                    vertices=v_out[:,step,...],
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,
                    translation_range=translation_range)

    return v_out, a_out, pos_out

def sample_n_steps_colors(
                        batch_size,
                        n_steps=2,
                        n_colors=10,
                        max_color_shift=2):
    a_out = np.zeros([batch_size,n_steps+1,1],dtype=int)
    # Sample initial colors:
    a_out[:,0,0] = np.random.randint(n_colors,size=[batch_size])

    if n_steps >= 1:
        # Sample other steps
        a_out[:,1:,0] = np.random.randint(
            max_color_shift*2+1,size=[batch_size,n_steps]) - max_color_shift
        pos_out = np.cumsum(a_out,axis=1)%n_colors
    else:
        # a_out = a_out[:,0]
        pos_out = a_out
    return a_out, pos_out


def get_image(vertices, triangles, figsize=(3,3), dpi=36, lim=1.5, col='white'):
    """
    Plots a 3D view of the object and returns it as a numpy array.
    """
    fig = plt.figure(figsize=figsize,dpi=dpi)   
    ax = plt.axes(projection='3d')

    ax.set_xlim([-lim,lim])
    ax.set_ylim([-lim,lim])
    ax.set_zlim([-lim,lim])

    ax.set_axis_off()

    x,y,z = vertices[:,0], vertices[:,1], vertices[:,2]
    ax.plot_trisurf(x,z,triangles,y,shade=True,color=col) 

    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image = image.reshape(figsize[0]*dpi,figsize[1]*dpi,3)
    # image /= 255
    # image = np.moveaxis(image,-1,0)
    plt.close()
    return image


def vertices_to_images(v, triangles, figsize=(3,3), dpi=24, lim=1.5,crop=0):
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
        lim (int):      Extent of the axis displayed is fixed to [-lim,lim]

    Returns:
        ndarray: a batch of images.

    """
    # v is of shape [batch, n_steps, n_vertices, 3]
    h,w = figsize[0]*dpi,  figsize[1]*dpi
    images_out = np.zeros(shape=[*v.shape[:-2],h,w,3])
    # v = v.reshape([-1,*v.shape[-2:]])
    for i in range(v.shape[0]):
        # if i%10 == 9:
        print(f'iter {i}/{v.shape[0]}')
        for j in range(v.shape[1]):
            images_out[i,j] = get_image(v[i,j], triangles,
                                        figsize=figsize, dpi=dpi, lim=lim)
    images_out /= 255
    images_out = np.moveaxis(images_out,-1,-3)
    if crop:
        images_out = images_out[...,crop:-crop,crop:-crop]
    return images_out

def vertices_to_colored_images(
                    v, triangles, color_idx, figsize=(3,3), dpi=24, lim=1.5,crop=0,n_colors=0):
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
        lim (int):      Extent of the axis displayed is fixed to [-lim,lim]

    Returns:
        ndarray: a batch of images.

    """
    colors_hsv = np.ones([n_colors,3])
    colors_hsv[:,0] = np.arange(n_colors)/n_colors
    colors_rgb = clr.hsv_to_rgb(colors_hsv)
    # v is of shape [batch, n_steps, n_vertices, 3]
    h,w = figsize[0]*dpi,  figsize[1]*dpi
    images_out = np.zeros(shape=[*v.shape[:-2],h,w,3])
    # v = v.reshape([-1,*v.shape[-2:]])
    for i in range(v.shape[0]):
        # if i%10 == 9:
        print(f'iter {i}/{v.shape[0]}')
        for j in range(v.shape[1]):
            images_out[i,j] = get_image(v[i,j], triangles,
                                        figsize=figsize, dpi=dpi, lim=lim,col=colors_rgb[color_idx[i,j,0]])
    images_out /= 255
    images_out = np.moveaxis(images_out,-1,-3)
    if crop:
        images_out = images_out[...,crop:-crop,crop:-crop]
    return images_out    

def generate_dataset(obj_filename, out_path, batch_size, figsize=(3,3), dpi=24, lim=1.5,
                     mode='continuous', n_values=0, 
                     rots_range=(-np.pi/2,np.pi/2),
                     rots_range_canonical=(-np.pi/2,np.pi/2),
                     rotate=False,
                     rotation_matrix_action=False,
                     translate=False,
                     translation_grid=3,
                     translation_stepsize=0.5,
                     translation_range=1,
                     n_steps=2, 
                     n_samples=10000, 
                     chunk_size=0, 
                     center=True, 
                     crop=0,
                     color=False,
                     n_colors=0,
                     max_color_shift=0,
                     attributes_dict={}):
    
    n_pos = 0 #number of dimensions for 3D position
    n_actions = 0
    if rotate:
        n_pos+=9 # flattened rotation matrix
        if rotation_matrix_action:
            n_actions+=9 # flattened rotation matrix
        else:
            n_actions+=3 # rotation angles roll yaw pitch
    
    if translate:
        n_actions+=3
        n_pos+=3
    
    if color:
        n_actions+=1
        n_pos+=1
    
    vertices, triangles = read_obj(obj_filename, center)
    
    with h5py.File(out_path, "w") as f:
        kwargs_images = {
            'dtype':np.float32,
            'shape' : (n_samples, n_steps+1, 3, figsize[0]*dpi-2*crop, 
                                                figsize[1]*dpi-2*crop), 
            'maxshape':(None, n_steps+1, 3, figsize[0]*dpi-2*crop, 
                                                figsize[1]*dpi-2*crop),
            }
        kwargs_actions = {
            'dtype':np.float32,
            'shape':(n_samples, n_steps+1, n_actions), 
            'maxshape':(None, n_steps+1, n_actions),
            }
        kwargs_pos = {
            'dtype':np.float32,
            'shape':(n_samples, n_steps+1, n_pos), 
            'maxshape':(None, n_steps+1, n_pos),
            }
        if chunk_size:
            kwargs_images['chunks'] = (chunk_size, n_steps+1, 3, 
                                figsize[0]*dpi-2*crop, figsize[1]*dpi-2*crop)
            kwargs_actions['chunks'] = (chunk_size, n_steps+1, n_actions)
            kwargs_pos['chunks'] = (chunk_size, n_steps+1, n_pos)
        
        dset_img = f.create_dataset('images', **kwargs_images)
        dset_rot = f.create_dataset('actions', **kwargs_actions)
        dset_pos = f.create_dataset('positions', **kwargs_pos)


        n_batches = n_samples//batch_size
        for i in range(n_batches):
            print(f'Sampling batch {i}/{n_batches}')
            if translate and rotate:
                v, a, pos = sample_n_steps_poses_from_canonical(
                    vertices, batch_size=batch_size,
                    n_steps=n_steps, mode=mode,
                    n_values=n_values, rots_range=rots_range,
                    rots_range_canonical=rots_range_canonical,
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,
                    translation_range=translation_range,
                    rotation_matrix_action=rotation_matrix_action)
            elif translate and not rotate:
                v, a, pos = sample_n_steps_trans_from_canonical(
                    vertices, batch_size=batch_size,
                    n_steps=n_steps,
                    translation_grid=translation_grid,
                    translation_stepsize=translation_stepsize,
                    translation_range=translation_range)
            elif rotate and not translate:
                v, a, pos = sample_n_steps_orientations_from_canonical(
                    vertices, batch_size=batch_size,
                    n_steps=n_steps, mode=mode,
                    n_values=n_values, rots_range=rots_range, 
                    rots_range_canonical=rots_range_canonical,
                    rotation_matrix_action=rotation_matrix_action)
            
            if color:
                a_cols, pos_cols = sample_n_steps_colors(
                        batch_size=batch_size,
                        n_steps=n_steps,
                        n_colors=n_colors,
                        max_color_shift=max_color_shift)
                a = np.concatenate([a,a_cols],axis=-1)
                pos = np.concatenate([pos,pos_cols],axis=-1)
                images = vertices_to_colored_images(
                    v, triangles, color_idx=pos_cols, figsize=figsize, 
                    dpi=dpi, lim=lim,crop=crop, n_colors=n_colors)
            else:
                images = vertices_to_images(
                    v, triangles, figsize=figsize, dpi=dpi, lim=lim,crop=crop)
            dset_img[i*batch_size:(i+1)*batch_size] = images
            dset_rot[i*batch_size:(i+1)*batch_size] = a
            dset_pos[i*batch_size:(i+1)*batch_size] = pos
        dset_img.attrs.update(attributes_dict)
    return 

def get_attributes_dict(obj_filename,  
                    figsize, 
                    dpi, 
                    lim,
                    mode, 
                    n_values, 
                    rots_range,
                    rots_range_canonical,
                    n_steps, 
                    n_samples,
                    center,
                    rotate,
                    rotation_matrix_action,
                    translate,
                    translation_grid,
                    translation_stepsize,
                    translation_range,
                    crop,
                    color,
                    n_colors,
                    max_color_shift):
    d = {
        "obj_filename":obj_filename,  
        "figsize":figsize,
        "dpi":dpi, 
        "lim":lim,
        "mode":mode, 
        "n_values":n_values, 
        "rots_range":rots_range,
        "rots_range_canonical":rots_range_canonical,
        "n_steps":n_steps, 
        "n_samples":n_samples,
        "center":center,
        "translate":translate,
        "rotate":rotate,
        "rotation_matrix_action":rotation_matrix_action,
        "translation_grid":translation_grid,
        "translation_stepsize":translation_stepsize,
        "translation_range":translation_range,
        "crop":crop,
        "color":color,
        "n_colors":n_colors,
        "max_color_shift":max_color_shift
    }
    return d

if __name__=='__main__':

    config = gargs.parse_cmd_arguments()
    if not os.path.exists(os.path.dirname(config.out_path)):
        os.makedirs(os.path.dirname(config.out_path))
    figsize = misc.str_to_ints(config.figsize)
    rots_range = misc.str_to_floats(config.rots_range)
    rots_range_canonical = misc.str_to_floats(config.rots_range_canonical)
    np.random.seed(config.gen_random_seed)

    attrs = get_attributes_dict(
                     obj_filename=config.obj_filename, 
                     figsize=figsize, 
                     dpi=config.dpi, 
                     lim=config.lim,
                     mode=config.mode, 
                     n_values=config.n_values, 
                     rots_range=rots_range,
                     rots_range_canonical=rots_range_canonical,
                     n_steps=config.n_steps, 
                     n_samples=config.n_samples,
                     center=config.center,
                     translate=config.translate,
                     rotate=config.rotate,
                     rotation_matrix_action=config.rotation_matrix_action,
                     translation_grid=config.translation_grid,
                     translation_stepsize=config.translation_stepsize,
                     translation_range=config.translation_range,
                     crop=config.crop,
                     color=config.color,
                     n_colors=config.n_colors,
                     max_color_shift=config.max_color_shift,
                     )
    
    generate_dataset(obj_filename=config.obj_filename, 
                     out_path=config.out_path, 
                     batch_size=config.batch_size, 
                     figsize=figsize, 
                     dpi=config.dpi, 
                     lim=config.lim,
                     mode=config.mode, 
                     n_values=config.n_values, 
                     rots_range=rots_range,
                     n_steps=config.n_steps, 
                     n_samples=config.n_samples,
                     center=config.center,
                     rotate=config.rotate,
                     rotation_matrix_action=config.rotation_matrix_action,
                     translate=config.translate,
                     translation_grid=config.translation_grid,
                     translation_stepsize=config.translation_stepsize,
                     translation_range=config.translation_range,
                     crop=config.crop,
                     attributes_dict=attrs,
                     color=config.color,
                     n_colors=config.n_colors,
                     max_color_shift=config.max_color_shift,
                     rots_range_canonical=rots_range_canonical
                     )