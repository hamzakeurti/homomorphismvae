import os
import sys

curr_dir = os.path.basename(os.path.abspath(os.curdir))
if curr_dir == 'supervised' and '..' not in sys.path:
    sys.path.insert(0, '..')
