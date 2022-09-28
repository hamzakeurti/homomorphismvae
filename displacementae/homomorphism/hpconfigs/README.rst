Hyperparameter Search configuration files.
==========================================

To run an hpsearch run the following command:

.. code-block:: console

    $ python3 -m hypnettorch.hpsearch.hpsearch --grid_module=hpconfigs.hpsearch_config_blockmlp_teapot_rots --run_cwd=$(pwd) --num_searches=40

replacing the --grid_module argument by the file name for the grid configuration. 