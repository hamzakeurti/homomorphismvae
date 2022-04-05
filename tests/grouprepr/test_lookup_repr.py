import unittest

import torch

from displacementae.grouprepr.lookup_representation \
    import LookupRepresentation

class TestLokupRepresentation(unittest.TestCase):
    def test_init(self):
        n_actions = 5
        n_repr_units = 3
        repr = LookupRepresentation(n_actions,n_repr_units)
        self.assertEqual(len(repr.action_reps),n_actions)
        self.assertEqual(list(repr.action_reps[0].shape),
                         [n_repr_units,n_repr_units])
        
    def test_forward(self):
        n_actions = 5
        n_repr_units = 3
        repr = LookupRepresentation(n_actions,n_repr_units)        
        batch_size = 10
        a = torch.randint(low=0,high=n_actions,size=(batch_size,))
        R = repr(a)
        self.assertEqual(list(R.shape),[batch_size,n_repr_units,n_repr_units])

    def test_act(self):
        n_actions = 5
        n_repr_units = 3
        repr = LookupRepresentation(n_actions,n_repr_units)        
        
        batch_size = 10
        a = torch.randint(low=0,high=n_actions,size=(batch_size,))
        z = torch.rand((batch_size,n_repr_units))
        z_out = repr.act(a,z)
        self.assertEqual(z_out.shape,z.shape)

if __name__ == '__main__':
    unittest.main()