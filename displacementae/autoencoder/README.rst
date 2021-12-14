Geometric Representation Learning through AutoEncoders
******************************************************

.. code-block:: console
    $ python3 train.py --dataset=dsprites --data_root=/home/hamza/datasets/dsprites --fixed_in_sampling=0,1,2,4,5 --fixed_values=0,2,5,14,14 --fixed_in_intervention=0,1,2,4,5 --lin_channels=128,64,32 --learn_geometry --plot_manifold_latent=0,1 --plot_vary_latents=3 --lr=0.0001 --epochs=100000 --use_cuda --variational --cuda_number=4 --beta=0.002 --batch_size=50 --num_train=500 --val_epoch=2000 --use_adam
