# Symmetry Based Representation Learning through Homomorphism AutoEncoders
---
Implementation associated with the paper [Homomorphism Autoencoder -- Learning Group Structured Representations from Observed Transitions](https://arxiv.org/abs/2207.12067).

The Homomorphism AutoEncoder HAE is a model trained on observed transitions $(o_t, g_t, o_{t+1})$ to jointly learn a group representation of the observed actions $g_t$ and a representation of the observations $o$.

Main scripts are provided in `./displacementae/homomorphism/`.

Best run commands are provided in the `./displacementae/homomorphism/README.rst`.


## Installation
---
The package can be installed by first building the package through:
```
 $ python setup.py sdist bdist_wheel
```
Then it can be installed in your environment through:
```
 $ pip install homomorphism-autoencoder-<VERSION>.tar.gz
```

## Datasets 
The Dsprites dataset and the 3D bunny .obj model need to be downloaded separately.

In addition, a transition dataset for the bunny dataset needs to be constructed from the downloaded .obj file using provided scripts in `./displacementae/data/obj3d/`.
Please refer to `./discplacementae/data/obj3d/README.md` for instructions.


## Documentation
---
Documentation can be built from the code's docstrings by following the 
instructions in `./docs/README.md`.