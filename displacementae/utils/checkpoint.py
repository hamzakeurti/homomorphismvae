import torch
import os

CKPT_FILENAME = 'checkpoint.tar'
EPOCH = 'epoch'
LOSSES = 'losses'
MODEL_STATE_DICT = 'model_state_dict'
OPTIMIZER_STATE_DICT = 'optimizer_state_dict'

def save_checkpoint(model,optimizer,losses,epoch,save_path):
    torch.save(
        {
            EPOCH:epoch,
            LOSSES:losses,
            MODEL_STATE_DICT:model.state_dict(),
            OPTIMIZER_STATE_DICT:optimizer.state_dict()
        },os.path.join(save_path,CKPT_FILENAME))

def load_checkpoint(model,optimizer=None,save_path=None):
    ckpt = torch.load(os.path.join(save_path,CKPT_FILENAME))
    model.load_state_dict(ckpt[MODEL_STATE_DICT])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt[OPTIMIZER_STATE_DICT])
    losses = ckpt[LOSSES]
    epoch = ckpt[EPOCH]
    return model,optimizer,losses,epoch
