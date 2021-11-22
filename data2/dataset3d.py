import torch
from torch.utils.data import Dataset, DataLoader,Sampler

from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
import os
import json
import numpy as np

from PIL import Image

SHAPES = ["Cube","Cylinder","Sphere","Capsule"]
FACTORS_IN_ORDER = ["shape","x_trans","y_trans","z_trans","theta_rot","phi_rot"]
NUMBER_VALUES = [4,5,5,3,6,7]
CONFIG = 'config.json'

class FactorIdx:
    SHAPE = 0
    X = 1
    Y = 2
    Z = 3
    THETA = 4
    PHI = 5

class Shapes:
    CUBE = 0
    CYLINDER = 1
    SPHERE = 2
    CAPSULE = 3

class Dataset3D(Dataset):

    def __init__(self,root:str,fixed_factors = [0],intervention = False,intervention_range = [-1,1]):
        super().__init__()
        self.root = root
        # TODO add sanity check of root path

        # TODO: Read from a json file in dataset root the lists: SHAPES FACTORS_IN_ORDER NUMBER_VALUES
        config_path = os.path.join(root,CONFIG)
        if os.path.exists(config_path):
            with open(config_path,'r') as f:
                configs = json.load(f)
            self.shapes = configs['shapes']
            self.num_values = configs['num_values']
        else:
            self.shapes = SHAPES
            self.num_values = NUMBER_VALUES
        
        self.cumprod = np.cumprod([1]+self.num_values[::-1])
        self.fixed_factors = fixed_factors
        self.intervention = intervention
        self.intervention_range = intervention_range
    
    def get_index(self,factors):
        index = 0
        base = 1
        for factor,num in reversed(list(enumerate(self.num_values))):
            index += factors[factor] * base
            base *= num
        return index 


    def get_filepath(self,factors):
        filename = '_'.join([f'{factors[i]}' for i in range(1,len(factors),1)]) + '.png'
        filepath = os.path.join(self.root,self.shapes[factors[0]],filename)
        return filepath


    def load_image(self,filepath):
        with Image.open(filepath) as image:
            return np.asarray(image,dtype=np.int32)

    def get_factors(self,i):
        factors = [None]*len(self.num_values)
        for k in range(len(self.num_values)):
            factors[k] = (i % self.cumprod[k+1]) // self.cumprod[k]
        return np.array(factors[::-1])


    def process_image(self,image):
        # WEIRD: WHY v1 dataset images have an additional color channel at 255
        if image.shape[-1] == 4:
            image = image[...,:3]
        image = np.moveaxis(image,-1,0) / 255.0

        return image

        
    def __getitem__(self,index):
        factors = self.get_factors(index)
        image = self.load_image(self.get_filepath(factors))
        image = self.process_image(image)
        if self.intervention:
            factors2,df = self.intervene(factors)
            image2 = self.load_image(self.get_filepath(factors2))
            image2 = self.process_image(image2)
            return image,factors,image2,factors2,df
        else:    
            return image,factors
    
    def __len__(self):
        return self.cumprod[-1]

    def intervene(self,factors_in,fixed_factors=None):
        if fixed_factors is None:
            fixed_factors = self.fixed_factors
        factors_out = np.copy(factors_in)
        df = np.empty_like(factors_in,dtype = int)
        for factor_id,num in enumerate(self.num_values):
            if factor_id not in fixed_factors:
                perturbation = np.random.randint(low=self.intervention_range[0],high= self.intervention_range[1]+1)
                factors_out[..., factor_id] = np.maximum(0,np.minimum(factors_in[..., factor_id] + perturbation,num-1))
                perturbation = factors_out[..., factor_id] - factors_in[...,factor_id]
                df[..., factor_id] = perturbation
            else:
                df[..., factor_id] = 0
        return factors_out,df

class FixedFactorSampler(Sampler):
    def __init__(self,fixed_factors,fixed_values,dataset=None,shuffle=False):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle
        self.fixed = fixed_factors
        self.vals = fixed_values
        self.facs = [None]*len(dataset.num_values)
        for i in range(len(self.fixed)):
            self.facs[self.fixed[i]] = self.vals[i]

        self.cumulative_product = np.concatenate([[1],np.cumprod(dataset.num_values[::-1])])

        self.cum_prod_fix = [1]
        self.n_samples = 1

        for f,num in reversed(list(enumerate(dataset.num_values))):
            if f not in self.fixed:
                self.n_samples *= num
                self.cum_prod_fix.append(self.n_samples)
        

    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        if not self.shuffle:
            return (self.get_index(i) for i in range(self.n_samples))
        else:
            import random
            return (self.get_index(i) for i in random.sample(list(range(self.n_samples)),self.n_samples))

    def get_index(self,i):
        """
        Transfers indices from range (0,self.n_samples) to indices of samples in the dataset with desired fixed factors.
        """
        ret = 0
        k = 0
        for f in range(len(self.dataset.num_values)-1,-1,-1):
            if self.facs[f] is not None:
                ret += self.facs[f]*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) * self.cumulative_product[::-1][f+1]
                k+=1
        return ret

def invert_and_cat(x1,z1,x2,z2,dz):
    inverse_batch = x2,z2,x1,z1,-dz
    out_batch = []
    for a,b in zip((x1,z1,x2,z2,dz),inverse_batch):
        out_batch.append(torch.cat((a,b)))
    return out_batch

#
#  
if __name__ == "__main__":
    # root = "/home/hamza/datasets/dataset3d/dataset3d/"
    # dataset = Dataset3D(root=root,intervention = True)
    # a = dataset[5]
    # dataloader = DataLoader(dataset,batch_size = 10)
    # for a in dataloader:
    #     print(a.shape)
    #     break


    root = "/home/hamza/datasets/dataset3d/v1/"

    # No intervention on  these
    fixed_factors = [0,4,5]

    # Constant in sampling
    constant_factors = [0,4,5]
    constant_values = [0,0,0]

    print('Loading data')
    dataset = Dataset3D(root,fixed_factors,intervention=True)
    print('\tData loaded')
    sampler = FixedFactorSampler(constant_factors,constant_values,dataset=dataset,shuffle=False)
    dataloader = DataLoader(dataset, batch_size=10,sampler = sampler)


    print(sampler.n_samples)
    # dataloader = DataLoader(dataset, batch_size=50,sampler=sampler)
    # for i,(x1,z1,x2,z2) in enumerate(dataloader):
    #     if (z1[:,-2-1] != dataset.factors_values[-2-1][1]).any():
    # #         print("NOOOO")
    for i in sampler:
        print(dataset[i][1])
