import unittest

import torch

from displacementae.grouprepr.soft_block_mlp_representation \
    import SoftBlockMLPRepresentation

class TestSoftBlockMLPRepresentation(unittest.TestCase):
    def test_init(self):
        n_action_units = 3
        dim = 5
        hidden_units=[10]
        repr = SoftBlockMLPRepresentation(n_action_units, dim, 
                 hidden_units,
                 normalize=False, 
                 device='cpu',
                 layer_norm=False, 
                 normalize_post_action=False,
                 exponential_map=True)
        self.assertEqual(list(repr.masks.shape),[dim-1,dim,dim])
        
    def test_get_masks(self):
        n_action_units = 3
        dim = 2
        hidden_units=[10]
        repr = SoftBlockMLPRepresentation(n_action_units, dim, 
                 hidden_units,
                 normalize=False, 
                 device='cpu',
                 layer_norm=False, 
                 normalize_post_action=False,
                 exponential_map=True)
        M = repr.masks
        M_exted = torch.zeros([1,2,2],dtype=bool)
        M_exted[0,1,0] =1
        M_exted[0,0,1] =1
        
        self.assertTrue((M==M_exted).all())

        n_action_units = 3
        dim = 4
        hidden_units=[10]
        repr = SoftBlockMLPRepresentation(n_action_units, dim, 
                 hidden_units,
                 normalize=False, 
                 device='cpu',
                 layer_norm=False, 
                 normalize_post_action=False,
                 exponential_map=True)
        M = repr.masks
        M_exted_0 = torch.ones([dim,dim],dtype=bool)
        M_exted_0[1:,1:] = 0
        M_exted_0[:1,:1] = 0
        
        self.assertTrue((M[0]==M_exted_0).all())
        
        M_exted_last = torch.ones([dim,dim],dtype=bool)
        M_exted_last[-1:,-1:] = 0
        M_exted_last[:-1,:-1] = 0
        self.assertTrue((M[-1]==M_exted_last).all())



if __name__ == '__main__':
    unittest.main()