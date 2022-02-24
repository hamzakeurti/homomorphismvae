Geometric Representation Learning through AutoEncoders
******************************************************

Only rotations on a heart
-------------------------

.. code-block:: console
 
 $ python3 train.py --dataset=dsprites --data_root=/home/hamza/datasets/dsprites --fixed_in_sampling=0,1,2,4,5 --fixed_values=0,2,5,14,14 --fixed_in_intervention=0,1,2,4,5 --lin_channels=128,64,32 --learn_geometry --plot_manifold_latent=0,1 --plot_vary_latents=3 --lr=0.0001 --epochs=100000 --use_cuda --variational --cuda_number=4 --beta=0.002 --batch_size=50 --num_train=500 --val_epoch=2000 --use_adam


3 cyclic displacements: rot, trans_x, trans_y
---------------------------------------------

Best run with scheduler toggling every 2 epochs:

.. code-block:: console
 
 $ python3 train.py --dataset=dsprites --data_root=/home/hamza/datasets/dsprites --fixed_in_sampling=0,1,2 --fixed_values=0,2,5 --fixed_in_intervention=0,1,2 --intervene --learn_geometry --conv_channels=32,32,32,32 --lin_channels=256,256 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --plot_manifold_latent=[0,1],[2,3],[4,5] --plot_vary_latents=[3],[4],[5] --lr=0.0005 --epochs=1001 --use_cuda --variational --cuda_number=0 --random_seed=43 --data_random_seed=43 --beta=1 --batch_size=1000 --val_epoch=1 --plot_epoch=100 --use_adam --n_free_units=0 --num_train=50000 --checkpoint --cyclic_trans