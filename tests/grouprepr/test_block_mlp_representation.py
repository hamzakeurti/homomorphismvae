import unittest

import torch

from displacementae.grouprepr.block_mlp_representation \
    import BlockMLPRepresentation

class TestBlockMLPRepresentation(unittest.TestCase):
    def test_init(self):
        n_action_units = 5
        n_repr_units = 10
        dims = [5,3,2]
        hidden_units=[]
        repr = BlockMLPRepresentation(n_action_units,n_repr_units,dims,
                                      hidden_units)
        self.assertEqual(repr.cumdims,[0,5,8,10])
        self.assertEqual(len(repr.subreps),3)
        subrepr_dim = [subrepr.dim_representation for subrepr in repr.subreps]
        self.assertEqual(subrepr_dim,dims)
        
    def test_forward(self):
        n_action_units = 5
        n_repr_units = 10
        dims = [5,3,2]
        hidden_units=[]
        repr = BlockMLPRepresentation(n_action_units,n_repr_units,dims,
                                      hidden_units)
        
        batch_size = 10
        a = torch.rand((batch_size,n_action_units))
        R = repr(a)
        self.assertEqual(list(R.shape),[batch_size,n_repr_units,n_repr_units])
        self.assertFalse(R[...,dims[0]:,:dims[0]].sum())


    def test_act(self):
        n_action_units = 5
        n_repr_units = 10
        dims = [5,3,2]
        hidden_units=[]
        repr = BlockMLPRepresentation(n_action_units,n_repr_units,dims,
                                      hidden_units)
        
        batch_size = 10
        a = torch.rand((batch_size,n_action_units))
        z = torch.rand((batch_size,n_repr_units))
        z_out = repr.act(a,z)
        self.assertEqual(list(z_out.shape),list(z.shape))

        R = repr(a)
        z_out_2 = torch.einsum('...ij,...j->...i',R,z)        
        self.assertTrue((z_out==z_out_2).all())



    