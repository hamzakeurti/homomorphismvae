import os
import pickle

SAVED_DIR = "/home/hamza/projects/displacementae/saved/"

def pickle_object(obj,save_dir,file_name):
    if file_name[-2:] != '.p':
        file_name+='.p'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(os.path.expanduser(os.path.join(save_dir,file_name)),'wb') as f:
        pickle.dump(obj,f)


def load_object(save_dir,file_name):
    if file_name[-2:] != '.p':
        file_name+='.p' 
    with open(os.path.expanduser(os.path.join(save_dir,file_name)),'rb') as f:
        return pickle.load(f)

