import torch

def get_distributions(model,dataloader,n_batch,device):
    zs = []
    labels = []
    with torch.no_grad():
        for i,batch in enumerate(dataloader):

            z = model.forward(batch[0].to(device),batch[-1].to(device))[1] # positional encoding is the second output of any model.
            if model.activate_latent is not None:
                z = model.activate_latent(z)    
            zs.append(z)
            labels.append(batch[1])
            print(i)
            if i>=n_batch:
                break
    zs2 = torch.vstack(zs).cpu().numpy()
    a = torch.vstack(labels).cpu().numpy()
    return zs2,a

