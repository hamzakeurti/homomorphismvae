import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset,Sampler
import os
import h5py

# Windows root: "D:\Projects\PhD\datasets\dsprites"

IMGS = "imgs"
LATENTS = "latents"
CLASSES = "classes"
VALUES = "values"

latents = ['shape','scale','orientation','pos_x','pos_y']
class LatentIdx:
    SHAPE = 0
    SCALE = 1 
    ORIENT = 2
    POSX = 3
    POSY = 4

class DspritesDataset(Dataset):
    """
    dsprites dataset
    """
    def __init__(self,root,intervene=False,displacement_range=[-1,1],immobile_joints = None,free_joints=None,hdf5=True):
        self.root = root
        self.intervene = intervene
        self.displacement_range = displacement_range
        self.n_joints = 5
        if immobile_joints is not None:
            if free_joints is not None:
                raise Exception("Only specify intervention joints or immobile joints, not both.")
            self.immobile_joints = immobile_joints
            self.free_joints = [i for i in range(self.n_joints) if i not in immobile_joints]
        else:
            if free_joints is not None:
                self.free_joints = free_joints
            else:
                if intervene:
                    self.free_joints = np.arange(5,dtype=int)
                else:
                    self.free_joints = []
            self.immobile_joints = [i for i in range(self.n_joints) if i not in self.free_joints]

        if hdf5:
            self._process_hdf5()

    def _process_hdf5(self):
        filepath = os.path.join(self.root,'dsprites.hdf5')
        self.file = h5py.File(filepath,'r')
        with h5py.File(filepath,'r') as f:
            self.images = self.file[IMGS][:]
            self.classes = self.file[LATENTS][CLASSES][:] # Remove color label, unique color over all dataset
            self.values = self.file[LATENTS][VALUES][:]
            self.num_classes = self.classes[-1][1:] + 1
        self.rotation_steps = 2*np.pi/ self.num_classes[LatentIdx.ORIENT]
    
    def __len__(self):
        return self.images.shape[0]
        
    def __getitem__(self, index):
        if self.intervene:
            index2,dj = self.f_intervene(index)
            return self._get_image(index),self._get_class(index),self._get_image(index2),self._get_class(index2),dj
        else:
            return self._get_image(index),self._get_class(index)

    def _get_image(self,index):
        return np.expand_dims(self.images[index],0).astype(np.float64)

    def _get_class(self,index):
        return self.classes[index][1:]

    def f_intervene(self,index):
        # intervention in the vicinity in the joints space 
        joints = self._get_class(index)
        #sample displacement
        if self.immobile_joints:
            len_dj = self.n_joints - len(self.immobile_joints)
        else:
            len_dj = self.n_joints
        dj = np.zeros(self.n_joints)
        dj[self.free_joints] = np.random.randint(low=self.displacement_range[0],high=self.displacement_range[1],size = len_dj)
        new_joints = joints
        new_joints,dj = self._intervene_linear(joints,dj)
        new_joints,dj = self._intervene_circular(new_joints,dj)
        i2 = self.get_index(new_joints)
        return i2,dj
    
    def _intervene_linear(self,joints,dj):
        new_joints = joints
        lin_idx = [LatentIdx.SCALE,LatentIdx.POSX,LatentIdx.POSY]
        new_joints[lin_idx] = np.clip(joints[lin_idx] + dj[lin_idx],0,self.num_classes[lin_idx])
        dj[lin_idx] = new_joints[lin_idx] - joints[lin_idx]
        return new_joints,dj
    
    def _intervene_circular(self,joints,dj):
        rot_idx = [LatentIdx.ORIENT]
        new_joints = joints
        new_joints[rot_idx] = (joints[rot_idx] + dj[rot_idx]) % self.num_classes[rot_idx]
        return new_joints,dj

    def get_index(self,joints):
        index = 0
        base = 1
        for j,joint in reversed(list(enumerate(joints))):
            index += joint * base
            base *= self.num_classes[j]
        return index

class FixedJointsSampler(Sampler):
    def __init__(self,fixed_joints,fixed_values,dataset=None,shuffle=False):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle
        self.fixed = fixed_joints
        self.vals = fixed_values
        self.facs = [None]*dataset.n_joints
        self.num_classes = dataset.num_classes
        for i in range(len(self.fixed)):
            self.facs[self.fixed[i]] = self.vals[i]
        

        self.cumulative_product = np.concatenate([[1],np.cumprod(self.num_classes[::-1])])

        self.cum_prod_fix = [1]
        
        self.n_samples = 1
        for f,num in reversed(list(enumerate(self.num_classes))):
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
        for f in range(len(self.num_classes)-1,-1,-1):
            if self.facs[f] is not None:
                ret += self.facs[f]*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) * self.cumulative_product[::-1][f+1]
                k+=1
        return ret
