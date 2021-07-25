import json
import os
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import optim

from models.autoencoder import VariationalOrthogonalAE,VariationalMixAE
from data import armeye
from data import dsprites
from utils import save,checkpoint


def parse_cmd_arguments(mode='orthogonal_vae'):
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str,
                        help='Json config file to load from,' +
                             'default are still filled according to argument parser.')
    # Data
    parser.add_argument('--dataset', type=str, default='armeye', 
                        help='Name of dataset')
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
    parser.add_argument('--model',type = int,default = 0,
                        help='choose model: \n0: othogonal vae\n1:mix vae')
    parser.add_argument('--learn_repr',type=bool,default=False,
                        help='Whether representation of action is learnable or static. Static corresponds to value chosen in --init_scale option')
    parser.add_argument('--repr_scale',type=int,default=0.1,
                        help='Scale of the unit action representation, and initial scale if --learn_repr is true')
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
    parser.add_argument('--n_samples', type=int, default=500,
                        help='Number of samples to visualize the latent distribution')
    # Optimizer
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    if mode == 'mix_vae':
        parser = _mixed_vae_arguments(parser)


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

def _mixed_vae_arguments(parser):
    parser.add_argument("--free_joints",type=list,default=None,
                        help="Specify joints/latents (ground truth labels) generating the visual data to act on")
    parser.add_argument("--rotation_idx",type=list,default = [], 
                        help='labels to intervene on with a rotation')
    parser.add_argument("--translation_idx",type=list,default = [], 
                        help='labels to intervene on with a translation')
    return parser


class Config:
    def __init__(self):
        pass

def load_config(id, save_dir='/home/hamza/projects/displacementae/saved/armeye/orthogonal_vae'):
    config = Config()
    with open(os.path.join(save_dir,str(id),'config.json'),'r') as f:
        config.__dict__ = json.load(f)
    return config

def setup_model_orthogonalvae(config):
    if config.intervene:
        n_latent = 2*(config.n_joints - len(config.immobile_joints))
        if n_latent == 0:
            raise Exception("intervene==True and all joints are immobile.")
    else:
        n_latent = config.latent_units
    model = VariationalOrthogonalAE(img_shape=config.img_shape,
                                    n_latent=n_latent, kernel_sizes=config.kernel_size,
                                    strides=config.strides, conv_channels=config.conv_channels, 
                                    hidden_units=config.hidden_units,intervene=config.intervene,
                                    device=config.device,learn_repr=config.learn_repr,
                                    repr_scale=config.repr_scale).double().to(config.device)
    return model

def setup_model_mixvae(config):
    model = VariationalMixAE(img_shape=config.img_shape,
                                    n_latent=config.latent_units, kernel_sizes=config.kernel_size,
                                    strides=config.strides, conv_channels=config.conv_channels, 
                                    hidden_units=config.hidden_units,intervene=config.intervene,
                                    rotation_idx = config.rotation_idx,translation_idx=config.translation_idx,
                                    device=config.device,
                                    learn_repr=config.learn_repr,repr_scale=config.repr_scale).double().to(config.device)
    return model

def setup_model(config):
    if config.model == 0:# orthogonal vae
        model = setup_model_orthogonalvae(config)
    if config.model == 1:
        model = setup_model_mixvae(config)
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
    if config.dataset == 'armeye':
        dataset,sampler = setup_data_armeye(config)
    elif config.dataset == 'dsprites':
        dataset,sampler = setup_data_dsprites(config)
    else:
        raise Exception(f'No support for dataset {config.dataset}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, sampler=sampler)
    return dataset, dataloader

def setup_data_armeye(config):
    dataset = armeye.ArmEyeDataset(config.data_root, n_joints=config.n_joints, intervene=config.intervene,
                            displacement_range=config.displacement_range, immobile_joints=config.immobile_joints)
    sampler = armeye.FixedJointsSampler(
        config.fixed_joints, config.fixed_values, dataset, shuffle=config.shuffle)
    return dataset,sampler

def setup_data_dsprites(config):
    dataset = dsprites.DspritesDataset(config.data_root,config.intervene,config.immobile_joints,config.free_joints)
    sampler = dsprites.FixedJointsSampler(config.fixed_joints, config.fixed_values, dataset, shuffle=config.shuffle)
    return dataset,sampler