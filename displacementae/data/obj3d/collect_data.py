import os, h5py
import numpy as np


os.chdir('/lustre/home/hkeurti/datasets/obj3d')

IMGS = 'images'
ACTS = 'actions'
POS = 'positions'

N_ACTS = 6
N_POS = 3
N_SAMPLES = 200000

BATCH = 200

clct_dir = os.path.join(os.getcwd(),'collect2')
prefix = 'teapot_debug'
suffix = '.hdf5'

with h5py.File('teapot_rots_trans_200k.hdf5','w') as fw:
    print('Created File to write')
    dset_imgs = fw.create_dataset(IMGS,shape=(N_SAMPLES,3,3,72,72),dtype=np.float32)
    dset_acts = fw.create_dataset(ACTS,shape=(N_SAMPLES,3,N_ACTS),dtype=np.float32)
    dset_pos = fw.create_dataset(POS,shape=(N_SAMPLES,3,N_POS),dtype=np.float32)
    print('Created datasets')
    for i in range(1000):
        with h5py.File(os.path.join(clct_dir,prefix+str(i)+suffix),'r') as fr:
            print(f'Opened file {os.path.join(clct_dir,prefix+str(i)+suffix)}')
            dset_imgs[(i)*BATCH:(i+1)*BATCH] = fr[IMGS][()]
            dset_acts[(i)*BATCH:(i+1)*BATCH] = fr[ACTS][()]
            dset_pos[(i)*BATCH:(i+1)*BATCH] = fr[POS][()]
            print(f'filled data between {i*BATCH} and {(i+1)*BATCH}')
print('Done')