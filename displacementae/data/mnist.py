import numpy as np
import torchvision
from torch.utils.data import Dataset
from skimage.transform import rotate

class MNISTDataset(Dataset):
    def __init__(self, root, rseed=None, intervene=True,
                angles_range=[-180,180], intervention_range=[-6,6], 
                num_train = 200, num_val=30, 
                num_digits = 100):
        super().__init__()

        self.train = True
        self.dataset = torchvision.datasets.MNIST(
            root= root, train = True, download = True)
        
        self.intervene = intervene

        # Random generator
        if rseed is not None:
            rand = np.random.RandomState(rseed)
        else:
            rand = np.random

        # Number of samples
        self.num_train = num_train
        self.num_val = num_val

        # Number of digits
        self.num_digits = num_digits

        # Sample digits
        self.digits_ids = rand.randint(len(self.dataset),size=self.num_digits)

        # Sample train set
        # Sample angles
        self.train_digits = rand.choice(self.digits_ids,size=self.num_train)
        self.train_angles_1 = rand.uniform(
            low=angles_range[0],high=angles_range[1],size=self.num_train)
        if self.intervene:
            self.train_angles_2 = rand.uniform(
                low=intervention_range[0],high=intervention_range[1],
                size=self.num_train)
            self.train_angles_2 += self.train_angles_1

        # sample test set
        self.val_digits = rand.choice(self.digits_ids,size=self.num_val)
        self.val_angles_1 = rand.uniform(
            low=angles_range[0],high=angles_range[1],size=self.num_val)
        if self.intervene:
            self.val_angles_2 = rand.uniform(
                low=intervention_range[0],high=intervention_range[1],
                size=self.num_val)
            self.val_angles_2 += self.val_angles_1


        data = {}
        data["in_shape"] = [1,28,28]
        data["action_shape"] = [int(intervene)]
        self._data = data

    def __len__(self):
        if self.train:
            return self.num_train
        else:
            return self.num_val

    def __getitem__(self, index):
        if self.train:
            angle1 = self.train_angles_1[index]
            idx = self.train_digits[index]
        else:
            angle1 = self.val_angles_1[index]
            idx = self.val_digits[index]
        img, label = self.dataset[idx]
        img1 = rotate(np.array(img),angle1)
        label1 = np.array([angle1,label])
        if self.intervene:
            if self.train:
                angle2 = self.train_angles_2[index]
            else:
                angle2 = self.val_angles_2[index]
            img2 = rotate(np.array(img),angle2)
            label2 = np.array([angle2,label])
            dj = angle2-angle1
            return img1,label1,img2,label2,dj
        else:
            return img1,label1,img1,label1,0
   
    @property
    def in_shape(self):
        return self._data["in_shape"]

    @property
    def action_shape(self):
        return self._data["action_shape"]

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self,value):
        self._train = value