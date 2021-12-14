import unittest
import numpy as np

import displacementae.data.armeye as armeye

root = "/home/hamza/datasets/armeye/transparent_small"

class TestArmEye(unittest.TestCase):

    def test_init(self):
        dhandler = armeye.ArmEyeDataset(
            root = root,fixed_in_sampling=[0,1],fixed_values=[0,5],
            intervene=True,fixed_in_intervention=[0],num_train=300,num_val=30)
        self.assertEqual(dhandler.intervened_on,[1,2])
        self.assertEqual(dhandler.n_joints,3)
        self.assertEqual(dhandler.fixed_in_intervention,[0])
        self.assertEqual(dhandler.varied_in_sampling,[2])
        
        self.assertEqual(len(dhandler.imshape),3)
        print(self.labels.shape)
    
if __name__ == '__main__':
    unittest.main()
        
