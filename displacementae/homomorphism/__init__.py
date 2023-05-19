import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
if curr_dir == 'homomorphism':
    if '..' not in sys.path:
        sys.path.insert(0, '..')
    if '../..' not in sys.path:
        sys.path.insert(0, '../..')
    