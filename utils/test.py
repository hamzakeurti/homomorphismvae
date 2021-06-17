import __init__
import torch
from utils import save
import os

def get_distributions(model,dataloader,n_samples,device):
    zs = []
    labels = []
    with torch.no_grad():
        for i,batch in enumerate(dataloader):
            if model.intervene:
                z = model.forward(batch[0].to(device),batch[-1].to(device))[1] # positional encoding is the second output of any model.
            else:
                z = model.forward(batch[0].to(device))[1]
            if model.activate_latent is not None:
                z = model.activate_latent(z)    
            zs.append(z)
            labels.append(batch[1])
            print(i)
            if i*zs[0].shape[0]>=n_samples:
                break
    zs2 = torch.vstack(zs).cpu().numpy()
    a = torch.vstack(labels).cpu().numpy()
    return zs2,a

def load_distribs(save_dir,id,epoch):
    loaded = save.load_object(os.path.join(save_dir,str(id)),f'distribs_{epoch}')
    labels = loaded['labels']
    latent = loaded['latent']
    return latent,labels

