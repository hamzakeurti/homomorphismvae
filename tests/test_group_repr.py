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
        