

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader,Sampler
import numpy as np
from matplotlib import image

class ArmEyeDataset(Dataset):
    """
    A pytorch dataset object to load samples from the arm eye dataset.
    The dataset is generated from a view of a simulated invisble robotic arm holding a visible object.
    Variation comes from joints movements.
    """
    LABELS = ['id','angle0','angle1','angle2','x','y','z']
    def __init__(self, root, intervene = False, intervention_range = [-1,1], 
                n_joints=3, rseed=None, fixed_in_sampling=[], 
                fixed_values=[],fixed_in_intervention=[], num_train=200, 
                num_val=30):
        super().__init__()

        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random

        self.num_train = num_train
        self.__num_val = num_val
        
        self.root = root
        self.labels_file = os.path.join(root,"labels.npy")
        self.labels = np.load(self.labels_file)
        self.n_joints = n_joints
        self.joints = np.arange(1,n_joints+1)
        self.pos_ids = np.arange(n_joints+1,n_joints+4)
        self.labels = self.process_labels()

        self.intervene = intervene
        self.n_joints = n_joints
        self.fixed_in_sampling = fixed_in_sampling
        self.fixed_values = fixed_values
        self.varied_in_sampling = np.array([i for i in self.joints \
            if i not in self.fixed_in_sampling])
        self.fixed_in_intervention = fixed_in_intervention
        self.intervened_on = np.array([i for i in self.joints \
            if i not in self.fixed_in_intervention])
        if not self.intervene:
            self.intervened_on = np.array([])
            self.fixed_in_intervention = self.joints


        data = {}
        data["in_shape"] = self.load_image(0).shape
        data["action_shape"] = [len(self.intervened_on)]
        self._data = data

        

    def process_labels(self):
        new_labels = np.empty_like(self.labels)
        new_labels[:,self.pos_ids] = self.labels[:,self.pos_ids]
        self.joint_steps = np.empty(self.n_joints)
        self.joint_n_vals = np.empty(self.n_joints,dtype=int)
        
        for i in range(self.n_joints):
            unique_vals,new_labels[:,self.joints[i]+1] = np.unique(self.labels[:,self.joints[i]+1],return_inverse=True)
            self.joint_steps[i] = unique_vals[1]-unique_vals[0]
            self.joint_n_vals[i] = len(unique_vals)
        return new_labels

    def __getitem__(self,i):
        if self.intervene:
            i2,dj = self.f_intervene(i)
            return self.load_image(i),self.labels[i],self.load_image(i2),self.labels[i2],dj
        else:
            img = self.load_image(i)
            return img,self.labels[i]

    def __len__(self):
        return self.labels.shape[0]

    def load_image(self,i):
        img_file = os.path.join(self.root,f'{i}.jpeg')
        img = image.imread(img_file)
        img = np.moveaxis(img,-1,0) / 255.0
        return img
    
    def f_intervene(self,i):
        # intervention in the vicinity in the joints space 
        joints = self.labels[i,self.joints_ids]
        #sample displacement
        if self.fixed_in_intervention:
            len_dj = self.n_joints - len(self.fixed_in_intervention)
        else:
            len_dj = self.n_joints
        dj = np.zeros(self.n_joints)
        dj[self.intervened_on] = self._rand.randint(
            low=self.intervention_range[0],high=self.intervention_range[1]+1,
            size = len_dj)
        new_joints = joints
        new_joints[self.intervened_on] = (joints[self.intervened_on] + dj) % self.joint_n_vals[self.intervened_on]
        i2 = self.get_index(new_joints)
        return i2,dj

    def get_index(self,joints):
        index = 0
        base = 1
        for j,num in reversed(list(enumerate(self.joint_n_vals))):
            index += joints[j] * base
            base *= num
        return int(index)
    
    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]


class FixedJointsSampler(Sampler):
    def __init__(self,fixed_joints,fixed_values,dataset=None,shuffle=False):
        super().__init__(None)
        self.dataset = dataset
        self.shuffle = shuffle
        self.fixed = fixed_joints
        self.vals = fixed_values
        self.facs = [None]*dataset.n_joints
        for i in range(len(self.fixed)):
            self.facs[self.fixed[i]] = self.vals[i]

        self.cumulative_product = np.concatenate([[1],np.cumprod(dataset.joint_n_vals[::-1])])

        self.cum_prod_fix = [1]
        
        self.n_samples = 1
        for f,num in reversed(list(enumerate(dataset.joint_n_vals))):
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
        for f in range(len(self.dataset.joint_n_vals)-1,-1,-1):
            if self.facs[f] is not None:
                ret += self.facs[f]*self.cumulative_product[::-1][f+1] 
            else:
                ret += ((i % self.cum_prod_fix[k+1]) // self.cum_prod_fix[k]) * self.cumulative_product[::-1][f+1]
                k+=1
        return ret

if __name__ == '__main__':
    pass

# if __name__ == '__main__':
#     root = os.path.expanduser('~/datasets/armeye/sphere_v1/transparent_small/')
#     dataset = ArmEyeDataset(root,n_joints=3,intervene=True,fixed_joints=[2])
#     dataloader = DataLoader(dataset,batch_size=50)
#     for x,y,x2,y2,dj in dataloader:
#         break
#     print(x.shape)