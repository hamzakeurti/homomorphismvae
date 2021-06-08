import torch
import torch.nn as nn
from torch import optim


import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import __init__
from experiments import train_utils
from utils import save, test

def kl_loss(mu, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)


def train(model, dataloader, optimizer, epoch, n_verbose=50):
    losses, rlosses, dlosses = [], [], []
    cumulated_loss, c_rloss, c_dloss = 0.0, 0.0, 0.0

    for i, (x1, z1, x2, z2, dj) in enumerate(dataloader):
        # to device
        device = model.steps.device
        a = (x1, z1, x2, z2, dj)
        x1, z1, x2, z2, dj = [e.to(device) for e in a]
        # -
        optimizer.zero_grad()

        # forward
        x2_hat, mu, logvar = model(x1.double(), dj.double())

        # loss
        rloss = nn.BCELoss(reduction='sum')(x2_hat, x2)
        dloss = kl_loss(mu, logvar)
        loss = rloss+dloss

        # backward
        loss.backward()

        # update
        optimizer.step()

        # log
        cumulated_loss += loss.item()
        c_rloss += rloss.item()
        c_dloss += dloss.item()

        if i % n_verbose == n_verbose - 1:
            losses.append(cumulated_loss / n_verbose)
            rlosses.append(c_rloss / n_verbose)
            dlosses.append(c_dloss / n_verbose)

            toprint = f'[{epoch}\t:{i+1}\t]\t'
            toprint = f'loss = {losses[-1]}\t'
            toprint += f'rloss = {rlosses[-1]}\t'
            toprint += f'dloss = {dlosses[-1]}\t'
            print(toprint)
            # print(f'[{epoch}\t:{i+1}\t]\tloss = {losses[-1]}')
            cumulated_loss, c_rloss, c_dloss = 0.0, 0.0, 0.0

        if i > config.max_iter:
            break
    return losses, rlosses, dlosses


if __name__ == "__main__":
    config = train_utils.parse_cmd_arguments()


    dataset, dataloader = train_utils.setup_data(config)
    model = train_utils.setup_model(config)
    start_epoch, end_epoch, losses, rlosses, dlosses = train_utils.setup_misc(config)

    free_joints = [i for i in range(config.n_joints) if i not in config.immobile_joints]
    model.steps = dataset.joint_steps[free_joints].to(config.device)
    
    path = os.path.join(config.save_path, config.id)


    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
 
    print('Starting training')

    model.train()
    for epoch in range(start_epoch, end_epoch):
        b_dist = ((epoch < 10) & (epoch % 2 == 0)) | (
            (epoch >= 10) & (epoch % 10 == 0))
        if b_dist:
            print(f'Getting distrib prior to epoch {epoch}')
            zs, a = test.get_distributions(
                model, dataloader, n_batch=20, device=config.device)
            save.pickle_object({"latent": zs, "labels": a}, os.path.join(
                config.save_path, config.id), f'distribs_{epoch}')
        print(f'Training epoch {epoch}')
        l = train(model, dataloader, optimizer, epoch, n_verbose=1)
        losses += l[0]
        rlosses += l[1]
        dlosses += l[2]
        save.pickle_object(model, os.path.join(config.save_path, config.id),'model')
        save.pickle_object({'total': losses, 'r': rlosses, 'd': dlosses}, os.path.join(
            config.save_path, config.id), 'losses3')