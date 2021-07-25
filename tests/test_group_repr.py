import unittest
from models import group_repr as grp

class TestAdaptiveRotation(unittest.TestCase):
    def test_init(self):
        orth = grp.AdaptiveRotationBlock(n_units=2)
        self.assertEqual(len(orth.unit_repr),1)
        orth = grp.AdaptiveRotationBlock(n_units=4)
        self.assertEqual(len(orth.unit_repr),2)
        
        orth = grp.AdaptiveRotationBlock(n_units=2,learn_repr=True)
        self.assertTrue(orth.unit_repr.requires_grad)
        orth = grp.AdaptiveRotationBlock(n_units=2,learn_repr=False)
        self.assertFalse(orth.unit_repr.requires_grad)
        
        orth = grp.AdaptiveRotationBlock(n_units=2,device='cpu')
        self.assertEqual(str(orth.unit_repr.device),'cpu')
        # orth = grp.AdaptiveRotationBlock(n_units=2,device='cuda:0')
        # self.assertEqual(str(orth.unit_repr.device),'cuda:0')
        
        self.assertIn(orth.unit_repr,orth.parameters())
