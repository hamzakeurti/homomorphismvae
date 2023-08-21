A documentation can be generated from the docstrings using sphinx as follows:


- install sphinx and sphinx-rtd-themes
```bash
conda install sphinx
conda install -c conda-forge sphinx_rtd_theme
```

- then from this directory run the following commands:
```bash
sphinx-quickstart
sphinx-apidoc -o ./source ../displacementae
make clean
make html
```

- The documentation can now be navigated through: `./build/index.html`.
