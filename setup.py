from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

# setup such that submodules are discovered from importing the top level module

setup(
    name='homomorphism-autoencoder',
    version='0.1.3',
    description='An Autoencoder that learns a group structured representation',
    long_description=long_description,
    # package_dir={'': 'displacementae'},
    packages=find_packages(),
    )
