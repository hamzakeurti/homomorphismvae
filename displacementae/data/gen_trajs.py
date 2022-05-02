from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import argparse
import h5py


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='teapot.obj',
                        help="Specify the path to dataset (teapot.obj/dsprite.hdf5)")
    parser.add_argument("--steps", type=int, default=4,
                        help="Steps per trajectory")
    parser.add_argument("--trajs", type=int, default=1000,
                        help="Number of trajectories to generate")
    parser.add_argument("--save_dir", type=str, default='./trajs/',
                        help="Path to save")
    parser.add_argument("--latent_active", type=int, nargs='+', default=[0, 1, 2, 3, 4],
                        help="Active latents (only for dsprite)")
    return parser.parse_args()


def gen_teapot(args):
    """
    Generates teapot trajectories with random policy
    The action space consists of 9 actions = 1 (idle) + 6 (+-rotations x/y/z) + 2 (+- color increment)
    code from https://github.com/IndustAI/learning-group-structure/
    teapot object from https://graphics.stanford.edu/courses/cs148-10-summer/as3/code/as3/teapot.obj
    """

    IMG_SIZE = 64

    triangles = []
    vertices = []
    with open(args.data) as f:
        for line in f:
            components = line.strip(' \n').split(' ')
            if components[0] == "f":  # face data
                # e.g. "f 1/1/1/ 2/2/2 3/3/3 4/4/4 ..."
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v":  # vertex data
                # e.g. "v  30.2180 89.5757 -76.8089"
                vertex = list(map(lambda c: float(c), components[1:]))
                vertices.append(vertex)
    vertices_init, triangles = np.array(vertices), np.array(triangles)

    angle = 2 * np.pi / 5
    colors = [
        [0, 0, 0],
        [255, 0, 0],
        [255, 255, 255],
        [0, 255, 0],
        [0, 0, 255]]

    matrices = [
        np.matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]),
        np.matrix([[np.cos(angle), 0, np.sin(angle)],
                   [0, 1, 0],
                   [-np.sin(angle), 0, np.cos(angle)]]),
        np.matrix([[np.cos(angle), 0, -np.sin(angle)],
                   [0, 1, 0],
                   [np.sin(angle), 0, np.cos(angle)]]),
        np.matrix([[1, 0, 0],
                   [0, np.cos(angle), np.sin(angle)],
                   [0, -np.sin(angle), np.cos(angle)]]),
        np.matrix([[1, 0, 0],
                   [0, np.cos(angle), -np.sin(angle)],
                   [0, np.sin(angle), np.cos(angle)]]),
        np.matrix([[np.cos(angle), np.sin(angle), 0],
                   [-np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]]),
        np.matrix([[np.cos(angle), -np.sin(angle), 0],
                   [np.sin(angle), np.cos(angle), 0],
                   [0, 0, 1]]),
        np.matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]),
        np.matrix([[1, 0, 0],
                   [0, 1, 0],
                   [0, 0, 1]]),
    ]
    imgs = []
    actions = []

    def vertices_to_img(v, c):
        # First, plot 3D image of a teapot and save as image

        x = np.asarray(vertices[:, 0]).squeeze()
        y = np.asarray(vertices[:, 1]).squeeze()
        z = np.asarray(vertices[:, 2]).squeeze()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.grid(None)
        ax.axis('off')
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_zlim([0, 3])
        ax.plot_trisurf(x, z, triangles, y, shade=True, color='white')
        ax.view_init(100, angle)
        img_path = os.path.join(traj_path, f'teapot_{i}.png')
        plt.savefig(img_path)
        plt.close()

        # Then load the image, crop, resize it, and change background color

        img = Image.open(img_path).convert('RGB')
        img = img.crop((100, 0, 350, 258))
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img)
        arr = np.where(arr == [255, 255, 255], colors[color_index], arr)
        return arr

    for t in tqdm(range(args.trajs)):
        traj_path = os.path.join(args.save_dir, str(t))
        os.makedirs(traj_path, exist_ok=True)

        # reset
        color_index = 0
        vertices = np.copy(vertices_init)
        imgs.append(vertices_to_img(vertices, color_index))

        # randomize initial state
        for _ in range(10):
            action = random.randrange(9)
            if action == 7:  # Change color by +1 increment
                color_index = (color_index + 1) % 5
            elif action == 8:  # Change color by -1 increment
                color_index = (color_index - 1) % 5
            elif action in [1, 2, 3, 4, 5, 6]:
                # Change viewpoint of teapot
                vertices = vertices * matrices[action]

        for i in range(args.steps):

            action = random.randrange(9)

            if action == 7:  # Change color by +1 increment
                color_index = (color_index + 1) % 5
            elif action == 8:  # Change color by -1 increment
                color_index = (color_index - 1) % 5
            elif action in [1, 2, 3, 4, 5, 6]:
                # Change viewpoint of teapot
                vertices = vertices * matrices[action]

            actions.append(action)

            arr = vertices_to_img(vertices, color_index)
            imgs.append(arr)

    imgs = np.array(imgs).reshape(args.trajs, args.steps + 1, IMG_SIZE, IMG_SIZE, 3)
    actions = np.array(actions).reshape(args.trajs, args.steps)

    # Save trajectories
    np.savez(os.path.join(args.save_dir, 'trajs.npz'),
             imgs=imgs,
             actions=actions,
             n_actions=np.array(9))


def gen_dsprites(args):
    """
    Generate trajectories from dsprites dataset

    Shape: square, ellipse, heart
    Scale: 6 values linearly spaced in [0.5, 1]
    Orientation: 40 values in [0, 2 pi]
    Position X: 32 values in [0, 1]
    Position Y: 32 values in [0, 1]
    """

    N_LATENTS = 5

    with h5py.File(args.data, 'r') as f:
        _imgs = f['imgs'][:]

    periods = [3, 6, 40, 32, 32]

    def coord_to_idx(c):
        return (32 * 32 * 40 * 6) * coord[0] \
            + (32 * 32 * 40) * coord[1] \
            + (32 * 32) * coord[2] \
            + 32 * coord[3] \
            + coord[4]

    imgs, actions = [], []
    for t in tqdm(range(args.trajs)):

        coord = [0] * N_LATENTS

        # randomize initial state
        for lat in args.latent_active:
            coord[lat] = random.randrange(periods[lat])

        imgs.append(_imgs[coord_to_idx(coord)])

        for i in range(args.steps):
            action = random.randrange(len(args.latent_active) * 2 + 1)

            if action != 0:
                _idx, delta = divmod(action - 1, 2)
                delta = delta * 2 - 1
                latent_idx = args.latent_active[_idx]
                coord[latent_idx] = (coord[latent_idx] + delta) % periods[latent_idx]
            imgs.append(_imgs[coord_to_idx(coord)])
            actions.append(action)

    imgs = np.array(imgs).reshape(args.trajs, args.steps + 1, 64, 64)
    actions = np.array(actions).reshape(args.trajs, args.steps)

    # Save trajectories
    np.savez(os.path.join(args.save_dir, 'trajs.npz'),
             imgs=imgs,
             actions=actions,
             n_actions=np.array(len(args.latent_active) * 2 + 1))


if __name__ == '__main__':

    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    if 'teapot' in args.data:
        print('Generating teapot trajectories...')
        gen_teapot(args)
    elif 'dsprites' in args.data:
        print('Generating dsprites trajectories...')
        gen_dsprites(args)
    else:
        raise ValueError
