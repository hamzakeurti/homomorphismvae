import json
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import optim

from models.autoencoder import VariationalOrthogonalAE
from data.armeye import ArmEyeDataset, FixedJointsSampler
from utils import save,checkpoint


def parse_cmd_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,
                        help='Json config file to load from,' +
                             'default are still filled according to argument parser.')
    # Data
    parser.add_argument('--n_joints', type=int, default=3,
                        help='Number of joints in the robot')
    parser.add_argument('--fixed_joints', type=int, nargs='+', default=[],
                        help='indices of fixed joints in sampling')
    parser.add_argument('--fixed_values', type=int, nargs='+',
                        default=[], help='Values of fixed joints')
    parser.add_argument('--immobile_joints', type=int, nargs='+',
                        default=[], help='Indices of fixed joints in sampling')
    parser.add_argument('--intervene', type=bool, default=True,
                        help='Whether to vary joint positions.')
    parser.add_argument('--shuffle', type=bool, default=True,
                        help='Shuffle the dataset.')
    parser.add_argument('--displacement_range', type=int, nargs=2,
                        default=[-3, 3], help='Range of uniform distribution from which to sample future joint position')
    parser.add_argument('--img_shape', type=int, nargs=2, default=[128, 128],
                        help='Shape of input images [h,w], ignore color channels,'+
                             'those should be added in convolution' + 
                             'channels as number of input channels.')
    parser.add_argument('--data_root', type=str, default="/home/hamza/datasets/armeye/sphere_v1/transparent_small/",
                        help='Root directory of the dataset relative from dataset directory.')
    # save
    parser.add_argument('--save_path', type=str, default='saved/armeye/orthogonal_vae/',
                        help='relative path from saved folder to save models and data for plots')
    parser.add_argument('--id', type=str,
                        help='relative path from saved save_path folder to save/load models and data for plots')
    parser.add_argument('--load_model', type=bool, default=False,
                        help='whether to load model from save_path')
    # Model
    parser.add_argument('--kernel_size', type=int, default=5,
                        help='kernel size of convolutional layers')
    parser.add_argument('--strides', type=int, default=1,
                        help='stride convolutional layers')
    parser.add_argument('--conv_channels', type=int, nargs='+',
                        default=[3, 16, 32, 32], help='channels convolutional layers')
    parser.add_argument('--hidden_units', type=int, nargs='?', const=[],
                        default=[50], help='hidden units')
    parser.add_argument('--latent_units',type=int,help='Number of latent units if --intervene is False')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    # Training
    parser.add_argument('--epochs', type=int, default=400,
                        help='number of epochs')
    parser.add_argument('--max_iter', type=int, default=400,
                        help='number of iterations')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Number of samples per training loop')
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    config = parser.parse_args()
    if config.json:
        with open(config.json, 'r') as f:
            json_config = json.load(f)
            config.__dict__.update(json_config)

    if not os.path.isabs(config.save_path):
        config.save_path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), config.save_path)

    if not config.json:
        path = os.path.join(config.save_path, config.id)
        config.__dict__['json'] = os.path.join(path, 'config.json')

    if not os.path.exists(os.path.join(config.save_path, config.id)):
        os.mkdir(os.path.join(config.save_path, config.id))

    with open(config.json, 'w') as f:
        json.dump(config.__dict__, f, sort_keys=False, indent=4)
    return config

class Config:
    def __init__(self):
        pass

def load_config(id, save_dir='/home/hamza/projects/displacementae/saved/armeye/orthogonal_vae'):
    config = Config()
    with open(os.path.join(save_dir,str(id),'config.json'),'r') as f:
        config.__dict__ = json.load(f)
    return config

def setup_model(config):
    path = os.path.join(config.save_path, config.id)
    device = config.device
    # if config.load_model:
    #     return save.load_object(path, 'model').double().to(device)
    if config.intervene:
        n_latent = 2*(config.n_joints - len(config.immobile_joints))
        if n_latent == 0:
            raise Exception("intervene==True and all joints are immobile.")
    else:
        n_latent = config.latent_units
    model = VariationalOrthogonalAE(img_shape=config.img_shape,
                                    n_latent=n_latent, kernel_sizes=config.kernel_size,
                                    strides=config.strides, conv_channels=config.conv_channels, 
                                    hidden_units=config.hidden_units,intervene=config.intervene,device=config.device).double().to(config.device)
    return model

def setup_misc(config):
    # path = os.path.join(config.save_path, config.id)
    # # if config.load_model:
    # #     # get epoch from saved data
    # #     fnames = os.listdir(path)
    # #     # parse fnames to get max saved epoch.
    # #     start_epoch = max([int(fname.split('_')[1][:-2])
    # #                        for fname in fnames if 'distribs' in fname])+1
    # #     # load losses
    # #     l = save.load_object(path, 'losses')
    # #     losses, rlosses, dlosses = l['total'], l['r'], l['d']
    start_epoch = 0
    losses, rlosses, dlosses = [], [], []
    end_epoch = start_epoch + config.epochs
    return start_epoch, end_epoch, losses, rlosses, dlosses

def setup_model_optimizer(config):
    model = setup_model(config)
    start_epoch, end_epoch, losses, rlosses, dlosses = setup_misc(config)
    optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
    if config.load_model:
        model,optimizer,l,epoch = checkpoint.load_checkpoint(
            model,optimizer,os.path.join(config.save_path,config.id))
        start_epoch = epoch
        end_epoch = start_epoch + config.epochs 
        losses, rlosses, dlosses = l['total'], l['r'], l['d']
    return model,optimizer,start_epoch, end_epoch, losses, rlosses, dlosses

def setup_data(config):
    dataset = ArmEyeDataset(config.data_root, n_joints=config.n_joints, intervene=config.intervene,
                            displacement_range=config.displacement_range, immobile_joints=config.immobile_joints)
    sampler = FixedJointsSampler(
        config.fixed_joints, config.fixed_values, dataset, shuffle=config.shuffle)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    # ,collate_fn=lambda x: [default_collate(a).to(config.device) for a in x])    
    return dataset, dataloader