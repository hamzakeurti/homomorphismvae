import os, h5py
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--collect_dir', type=str, help='Name of directory containing batch images.')
parser.add_argument('--out', type=str, help='Output hdf5 file name with extension.')
parser.add_argument('--store_pos', action='store_true', help='Whether datasets also contain position data.' )
parser.add_argument('--n_acts', type=int, default=3, help='Number of action elemnts.')
parser.add_argument('--n_pos', type=int, default=3, help='Number of position elemnts.')
parser.add_argument('--batch_size', type=int, default=300, help='Number of samples per batch.')
parser.add_argument('--n_batches', type=int, default=1000, help='Number of batches.')
parser.add_argument('--prefix', type=str, default='bunny', help='prefix of batch files. This is followed by the batch index.')

config = parser.parse_args()

os.chdir('/lustre/home/hkeurti/datasets/obj3d')

IMGS = 'images'
ACTS = 'actions'
POS = 'positions'

N_ACTS = config.n_acts
N_POS = config.n_pos


BATCH = config.batch_size
N_BATCHES = config.n_batches
N_SAMPLES = BATCH * N_BATCHES



clct_dir = os.path.join(os.getcwd(),config.collect_dir)
prfix = config.prefix
suffix = '.hdf5'

with h5py.File(config.out,'w') as fw:
    print('Created File to write')
    dset_imgs = fw.create_dataset(IMGS,shape=(N_SAMPLES,3,3,72,72),dtype=np.float32)
    dset_acts = fw.create_dataset(ACTS,shape=(N_SAMPLES,3,N_ACTS),dtype=np.float32)
    if config.store_pos:
        dset_pos = fw.create_dataset(POS,shape=(N_SAMPLES,3,N_POS),dtype=np.float32)
    print('Created datasets')
    for i in range(N_BATCHES):
        with h5py.File(os.path.join(clct_dir,prfix+str(i)+suffix),'r') as fr:
            print(f'Opened file {os.path.join(clct_dir,prfix+str(i)+suffix)}')
            dset_imgs[(i)*BATCH:(i+1)*BATCH] = fr[IMGS][()]
            dset_acts[(i)*BATCH:(i+1)*BATCH] = fr[ACTS][()]
            if config.store_pos:
                dset_pos[(i)*BATCH:(i+1)*BATCH] = fr[POS][()]
            print(f'filled data between {i*BATCH} and {(i+1)*BATCH}')
            if i==0:
                dset_imgs.attrs.update(dict(fr[IMGS].attrs))
print('Done')