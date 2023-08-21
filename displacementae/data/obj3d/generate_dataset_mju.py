import __init__

import os, platform
import h5py
import argparse
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple, Union
from tqdm import tqdm
if platform.system() == 'Linux':
    os.environ['MUJOCO_GL']='egl' 
import mujoco

os.environ['HDF5_USE_FILE_LOCKING']='FALSE' 

from displacementae.data.obj3d.world_model import WorldModel
from displacementae.utils import misc

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate a dataset of object transitions.")
    parser.add_argument("--object_dir", type=str, 
                        help="Directory containing .obj files.")
    parser.add_argument("--object_name", type=str,
                        help="Name of the object to be rendered. Needs to " + 
                             " be the same as .obj file")
    parser.add_argument("--output_path", type=str, 
                        help="Path to save the dataset.")
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Number of samples to generate.')
    parser.add_argument('--n_steps', type=int, default=1,
                        help='Number of action steps per sample.')
    parser.add_argument('--rotation_range', type=str, default=f"0,{np.pi/4}",
                        help='Range of rotations in radians.')
    parser.add_argument('--rotation_format', type=str, default='quat',
                        choices=['quat', 'mat', 'euler'],
                        help='Format of the rotations.')
    parser.add_argument('--color', action='store_true',
                        help='Whether to act on color or not.')
    parser.add_argument('--continuous_color',action='store_true',
                        help='Whether hues are continuously sampled.')
    parser.add_argument('--color_range', type=str, default=None,
                        help='Range of color displacements. ' + 
                        'If --continuous_color, this is a float between ' + 
                        '0 and 1. With 1 corresponding to the whole ' + 
                        'color wheel.')
    parser.add_argument('--n_colors', type=int, default=None,
                        help='Number of equally spaced colors on the hue ' +
                             'wheel.')
    parser.add_argument('--figsize', type=str, default='72,72',
                        help='Size of the rendered image.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed.')
    parser.add_argument('--verbose', action='store_true',
                        help='Whether to print progress.')
    parser.add_argument('--write_batch_size', type=int, default=200,
                        help='Size of the batches of samples written at the ' +
                             'same time on the hdf5 file.')
    
    args = parser.parse_args()
    return args


def generate_transitions(world_model:WorldModel, n_samples:int, n_steps:int, 
                         rot_range:Tuple[float, float], 
                         rotation_format:str='quat', color:bool=False,
                         continuous_color:bool=False, 
                         n_colors:Optional[int]=None, 
                         color_range:Optional[Union[int,float]]=None
                         ) -> Tuple[npt.NDArray[np.float32], 
                                    npt.NDArray[np.float32], 
                                    npt.NDArray[np.float32]]:
    """
    Generate a dataset of transitions (image, action, image,...) .
    """

    n_color_params = 1 if color else 0
    # Initialize arrays
    n_rotation_params = get_n_rotation_params(rotation_format) 
    n_actions = n_rotation_params + n_color_params
    n_poses = n_actions
    images = np.zeros((n_samples, n_steps+1, 3, *world_model.figsize), 
                      dtype=np.uint8)
    actions = np.zeros((n_samples, n_steps, n_actions), dtype=np.float32)
    poses = np.zeros((n_samples, n_steps+1, n_poses), dtype=np.float32)

    # Sample initial orientations (euler angles).
    euler = np.random.uniform(-np.pi, np.pi, size=(n_samples, 3))
    quat = misc.euler_to_quat(euler)

    # sample rotations
    deuler = np.random.uniform(rot_range[0], rot_range[1], 
                               size=(n_samples, n_steps, 3))
    dquat = misc.euler_to_quat(deuler)

    if rotation_format == 'euler':
        poses[:, 0, :n_rotation_params] = euler
        actions[:, :, :n_rotation_params] = deuler
    elif rotation_format == 'quat':
        poses[:, 0, :n_rotation_params] = quat
        actions[:, :, :n_rotation_params] = dquat
    elif rotation_format == 'mat':
        poses[:,0,:n_rotation_params] = misc.quat_to_mat(quat.astype(np.float64)).astype(np.float32)
        actions[:,:,:n_rotation_params] = misc.quat_to_mat(dquat.astype(np.float64)).astype(np.float32)

    if color:
        if (not continuous_color and n_colors is None) or color_range is None:
            raise ValueError("color_range and at least one of n_colors or " +
                             "continuous_colors must be specified when " + 
                             "color is True.")
        if not continuous_color:
            assert n_colors is not None
            assert isinstance(color_range,int)
            hues = np.arange(n_colors)/n_colors # hue colors from 0 to 1
            colors_rgb = misc.hue_to_rgb(hues)
            colors_rgba = np.concatenate([colors_rgb, np.ones([n_colors,1])], axis=1)
            # sample initial colors
            poses[:, 0, -1] = np.random.randint(0, n_colors, size=n_samples)
            # sample color displacements
            actions[:, :, -1] = np.random.randint(-color_range, color_range+1,
                                                    size=(n_samples, n_steps))
            poses[:, 1:, -1] = actions[:, :, -1]
            poses[:, :, -1] = np.cumsum(poses[:, :, -1], axis=1) % n_colors
        else:
            assert isinstance(color_range,float)
            # sample initial hues
            poses[:, 0, -1] = np.random.random(size=n_samples)
            # sample hue displacements
            actions[:, :, -1] = (np.random.random(size=(n_samples, n_steps)) * 2 - 1)  * color_range
            # compute hues for next steps
            poses[:, 1:, -1] = actions[:, :, -1]
            poses[:, :, -1] = np.cumsum(poses[:, :, -1], axis=1) % 1
            # convert hues to rgba
            colors_rgba = np.ones([n_samples,n_steps+1,4])
            colors_rgba[...,:3] = misc.hue_to_rgb(poses[:,:,-1])
            

    for i in tqdm(range(n_samples)):
        # Sample initial color
        # if color:
        #     color = np.random.randint(0, n_colors)
        #     poses[i, 0, 4] = color

        # Sample initial image
        world_model.set_orientation(quat[i])
        if color:
            if not continuous_color:
                world_model.set_color(colors_rgba[int(poses[i, 0, -1])])
            else:
                world_model.set_color(colors_rgba[i, 0])

        images[i, 0] = np.moveaxis(world_model.render(), -1, 0)

        # Sample subsequent images
        for j in range(n_steps):
            # Update pose
            world_model.rotate_by(dquat[i, j])
            quat_pose = world_model.orientation
            if rotation_format == 'euler':
                poses[i, j+1, :3] = misc.quat_to_euler(quat_pose)
            elif rotation_format == 'quat':
                poses[i, j+1, :4] = quat_pose
            elif rotation_format == 'mat':
                poses[i, j+1, :9] = misc.quat_to_mat(quat_pose.astype(np.float64)).astype(np.float32)
            
            if color:
                if not continuous_color:
                    world_model.set_color(colors_rgba[int(poses[i, j+1, -1])])
                else:
                    world_model.set_color(colors_rgba[i, j+1])

            # Update image
            images[i, j+1] = np.moveaxis(world_model.render(), -1, 0)

    images = images/255.0
    return images, actions, poses


def get_n_rotation_params(rotation_format:str) -> int:
    if rotation_format == 'quat':
        return 4 
    elif rotation_format == 'euler':
        return 3
    elif rotation_format == 'mat':
        return 9
    else:
        raise ValueError(f"Unknown rotation format: {rotation_format}")
        

def main():
    """
    Generate a dataset.
    """
    args = parse_args()
    np.random.seed(args.seed)
    if args.verbose:
        print('Generating dataset...')
    if args.verbose:
        print('Parsing arguments...')
    figsize = tuple(misc.str_to_ints(args.figsize))
    rots_range = tuple(misc.str_to_floats(args.rotation_range))
    if args.verbose:
        print('Creating world model...')
    world_model = WorldModel(args.object_dir, args.object_name, 
                             figsize=figsize)
    if args.verbose:
        print('Generating transitions...')
    
    color_rng = None
    if args.color:
        color_rng = float(args.color_range) if args.continuous_color \
                 else int(args.color_range)

    imgs, acts, poses = generate_transitions(
                world_model, n_samples=args.n_samples, n_steps=args.n_steps,
                rot_range=rots_range, rotation_format=args.rotation_format,
                color=args.color, continuous_color=args.continuous_color, 
                n_colors=args.n_colors, color_range=color_rng)
    
    n_color_params = 1 if args.color else 0
    # Initialize arrays
    n_rotation_params = get_n_rotation_params(args.rotation_format) 
    n_actions = n_rotation_params + n_color_params


    with h5py.File(os.path.expanduser(args.output_path), 'w') as f:
        if args.verbose:
            print('Saving in HDF5...')
        kwargs_images = {
            'dtype':np.float32,
            'shape' : (args.n_samples, args.n_steps+1, 3, figsize[0], 
                                                figsize[1]), 
            'maxshape':(None, args.n_steps+1, 3, figsize[0], 
                                                figsize[1]),
        }
        kwargs_actions = {
            'dtype':np.float32,
            'shape' : (args.n_samples, args.n_steps, n_actions), 
            'maxshape':(None, args.n_steps, n_actions),
        }
        kwargs_pos = {
            'dtype':np.float32,
            'shape' : (args.n_samples, args.n_steps+1, n_actions), 
            'maxshape':(None, args.n_steps+1, n_actions),
        }

        f.attrs['n_samples'] = args.n_samples
        f.attrs['n_steps'] = args.n_steps
        f.attrs['rotation_range'] = args.rotation_range
        f.attrs['color'] = args.color
        if args.color:
            f.attrs['color_range'] = color_rng
            f.attrs['continuous_color'] = args.continuous_color
            f.attrs['n_colors'] = 0 if args.continuous_color else args.n_colors
        f.attrs['figsize'] = args.figsize
        f.attrs['seed'] = args.seed
        f.attrs['rotation_format'] = args.rotation_format

        f.attrs['mode'] = 'continuous'
        f.attrs['translate'] = False
        f.attrs['rotate'] = True

        dset_imgs = f.create_dataset('images', **kwargs_images)
        dset_act = f.create_dataset('actions', **kwargs_actions)
        dset_pos = f.create_dataset('positions', **kwargs_pos)

        bsize = args.write_batch_size
        n_batches = args.n_samples // bsize
        for i in tqdm(range(n_batches)):
            dset_imgs[i*bsize:(i+1)*bsize] = imgs[i*bsize:(i+1)*bsize]
            dset_act[i*bsize:(i+1)*bsize] = acts[i*bsize:(i+1)*bsize]
            dset_pos[i*bsize:(i+1)*bsize] = poses[i*bsize:(i+1)*bsize]

if __name__ == '__main__':
    main()