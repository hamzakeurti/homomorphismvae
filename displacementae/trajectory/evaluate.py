import __init__
import displacementae.utils.checkpoint as ckpt
import displacementae.utils.plotting_utils as plt_utils
import displacementae.networks.variational_utils as var_utils
import displacementae.networks.network_utils as net_utils
from displacementae.data.trajs import TrajectoryDataset
import displacementae.utils.args as uargs
from displacementae.grouprepr.representation_utils import Representation

import argparse
import os

import torch
from torch.utils.data import DataLoader

import numpy as np


if __name__ == '__main__':
    rep = Representation.BLOCK_LOOKUP

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data', type=str)
    parser.add_argument('--data_type', type=str, default='dsprites')
    parser.add_argument('--out_dir', type=str, default='./out/')
    parser.add_argument('--plot_on_black', default=False, action='store_true')
    uargs.net_args(parser)
    uargs.group_repr_args(parser, rep)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dhandler = TrajectoryDataset(args.data, args.data_type)
    dloader = DataLoader(dataset=dhandler, batch_size=10, shuffle=False)

    nets = net_utils.setup_network(args, [dhandler, ], device, mode='trajectory',
                                   representation=rep)
    nets, _, _, _ = ckpt.load_checkpoint(nets, save_path=args.checkpoint)
    nets.eval()

    total_losses, step_losses, bce_losses, kl_losses = [], [], [], []
    with torch.no_grad():
        for batch in dloader:
            imgs, actions = [torch.as_tensor(elem, dtype=torch.float64, device=device) for elem in batch]
            # imgs is of shape
            # [batch_size, n_steps+1, channels, height, width]
            # actions is of shape [batch_size, n_steps, n_actions]
            x1 = imgs[:, 0]  # initial observed image
            xi = imgs[:, 1:]

            ### Forward ###
            h, mu, logvar = nets.encode(x1)
            h_hat = nets.act(h, actions)
            xi_hat = torch.sigmoid(nets.decode(h_hat))


            ### Losses
            # Reconstruction
            bce_loss_elementwise = var_utils.bce_loss(xi_hat, xi, 'none')
            bce_loss_step = bce_loss_elementwise.sum(dim=[2, 3, 4])
            bce_loss = bce_loss_step.mean(dim=1)
            total_loss = bce_loss

            bce_losses.append(bce_loss)
            total_losses.append(total_loss)
            step_losses.append(bce_loss_step)

    bce_losses = torch.cat(bce_losses).mean()
    total_loss = torch.cat(total_losses).mean()
    step_loss = torch.cat(step_losses)

    print(f'loss: {total_loss.item():.2f}')

    # Save Losses
    np.save(os.path.join(args.out_dir, 'step_loss'),
            step_loss.cpu().numpy())

    plt_utils.plot_step_recon_loss(step_loss.cpu().numpy(), args, args.out_dir)
