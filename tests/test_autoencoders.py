import unittest
from models import autoencoder as ae


class TestVariationalOrthogonalAE(unittest.TestCase):
    def test_init(self):
        model = ae.VariationalOrthogonalAE(img_shape=[64,64],n_latent=2,kernel_sizes=9,strides=2,conv_channels=[3,5],hidden_units=[],learn_repr=True)
        self.assertTrue(model.orthogonal.unit_repr in model.parameters())