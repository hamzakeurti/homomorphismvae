import torch
import torch.nn as nn


import os

import __init__
from experiments import train_utils
from utils import save, test,checkpoint

def kl_loss(mu, logvar):
    return 0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1. - logvar)


def train(model, dataloader, optimizer, epoch, n_verbose=50):
    losses, rlosses, dlosses = [], [], []
    cumulated_loss, c_rloss, c_dloss = 0.0, 0.0, 0.0

    for i, batch in enumerate(dataloader):
        
        # to device
        device = model.steps.device
        batch = [e.to(device).double() for e in batch]
        # -
        optimizer.zero_grad()

        # forward
        if model.intervene:
            x1, j1, x2, j2, dj = batch
            out, mu, logvar = model(x1, dj)
            target = x2
        else:
            x1,j1 = batch
            out, mu, logvar = model(x1)
            target = x1
        
        # loss
        rloss = nn.BCELoss(reduction='sum')(out, target)
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
            toprint += f'loss = {losses[-1]}\t'
            toprint += f'rloss = {rlosses[-1]}\t'
            toprint += f'dloss = {dlosses[-1]}\t'
            print(toprint)
            # print(f'[{epoch}\t:{i+1}\t]\tloss = {losses[-1]}')
            cumulated_loss, c_rloss, c_dloss = 0.0, 0.0, 0.0

        if i > config.max_iter:
            break
    return losses, rlosses, dlosses

if __name__ == "__main__":
    config = train_utils.parse_cmd_arguments(mode='mix_vae')

    path = os.path.join(config.save_path, config.id)

    dataset, dataloader = train_utils.setup_data(config)

    model,optimizer,start_epoch, end_epoch, losses, rlosses, dlosses = train_utils.setup_model_optimizer(config)
    model.steps = torch.tensor(dataset.rotation_steps).to(config.device)

    model.train()
    for epoch in range(start_epoch, end_epoch):
        b_dist = ((epoch < 10) & (epoch % 2 == 0)) | (
            (epoch >= 10) & (epoch % 10 == 0))
        if b_dist:
            print(f'Getting distrib prior to epoch {epoch}')
            zs, a = test.get_distributions(
                model, dataloader, n_samples=config.n_samples, device=config.device)
            save.pickle_object({"latent": zs, "labels": a}, os.path.join(
                config.save_path, config.id), f'distribs_{epoch}')
        print(f'Training epoch {epoch}')
        l = train(model, dataloader, optimizer, epoch, n_verbose=config.verbose)
        losses += l[0]
        rlosses += l[1]
        dlosses += l[2]
        save.pickle_object(model, os.path.join(config.save_path, config.id),'model')
        save.pickle_object({'total': losses, 'r': rlosses, 'd': dlosses}, os.path.join(
            config.save_path, config.id), 'losses3')
        checkpoint.save_checkpoint(model,optimizer,losses={'total': losses, 'r': rlosses, 'd': dlosses},epoch=epoch,save_path = path)

