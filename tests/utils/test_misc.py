import unittest
import numpy as np
import displacementae.utils.misc as misc

class TestConversions(unittest.TestCase):
    def test_str_to_ints(self):
        a = '3,4'
        ret = misc.str_to_ints(a)
        self.assertEqual(ret,[[3,4]])

        a = '[3,4],[5]'
        ret = misc.str_to_ints(a)
        self.assertEqual(ret,[[3,4],[5]])


if __name__ == '__main__':
    unittest.main()
        
