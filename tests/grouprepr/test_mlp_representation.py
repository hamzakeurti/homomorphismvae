import unittest

import torch

from displacementae.grouprepr.mlp_representation \
    import MLPRepresentation

class TestMLPRepresentation(unittest.TestCase):
    def test_init(self):
        n_action_units = 5
        n_repr_units = 3
        hidden_units=[]
        repr = MLPRepresentation(n_action_units,n_repr_units,
                                      hidden_units)
        self.assertEqual(list(repr.parameters()),list(repr.net.parameters()))
        
    def test_forward(self):
        n_action_units = 5
        n_repr_units = 3
        hidden_units=[]
        repr = MLPRepresentation(n_action_units,n_repr_units,
                                      hidden_units)
        
        batch_size = 10
        a = torch.rand((batch_size,n_action_units))
        R = repr(a)
        self.assertEqual(list(R.shape),[batch_size,n_repr_units,n_repr_units])

    def test_act(self):
        n_action_units = 5
        n_repr_units = 3
        hidden_units=[]
        repr = MLPRepresentation(n_action_units,n_repr_units,
                                      hidden_units)
        
        batch_size = 10
        a = torch.rand((batch_size,n_action_units))
        z = torch.rand((batch_size,n_repr_units))
        z_out = repr.act(a,z)
        self.assertEqual(z_out.shape,z.shape)

if __name__ == '__main__':
    unittest.main()