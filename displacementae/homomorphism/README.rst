=========================
Homomorphism AutoEncoders
=========================

==================================
Structured Representation Learning
==================================

Single Shape
------------

For learning a structured representation of the ellipse dataset acted on by the group of cyclic translations:
### Using the Block MLP Group Representation:

.. code-block:: console

 $ python3 train_block_mlp_repr.py --dataset=dsprites --data_root=<PATH-TO-DATASET> --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,1,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=10000 --batch_size=500 --epochs=101 --lr=0.001 --toggle_training_every=2,2 --shuffle=1 --use_adam --use_cuda --conv_channels=64,64,64,64 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --dims=2,2 --group_hidden_units=128,128 --reconstruct_first --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_pca --plot_vary_latents=[4,5]

### Using the Sparse Block MLP Group Representation:

.. code-block:: console

 $ python3 train_soft_block_mlp_repr.py --dataset=dsprites --data_root=<PATH-TO-DATASET> --kernel_sizes=6,4,4,4 --conv_channels=32,32,32,32  --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,1,2,3 --fixed_values=0,2,5,0 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=0 --num_train=50000 --batch_size=500 --epochs=201 --lr=0.001 --toggle_training_every=6,4 --shuffle=1 --use_adam --use_cuda --strides=2,2,1,1 --lin_channels=1024 --beta=0 --net_act=relu --latent_loss_weight=400 --latent_loss --reconstruct_first --group_hidden_units=1024 --dim=8 --grp_loss_on --grp_loss_weight=0.1 --varphi_units=200,50 --varphi_random_seed=10 --varphi_act=relu --val_epoch=10 --num_val=500 --plot_epoch=10 --plot_manifold_latent=[0,1] --plot_pca --plot_manifold --plot_matrices --plot_reconstruction --plot_vary_latents=[4,5] --out_dir=<OUTPUT-DIRECTORY>


Multiple Shapes
---------------

For learning a structured representation in the non transitive case where thedataset contains all three shapes acted on by the group pf translation:

.. code-block:: console

 $ python3 train_block_mlp_repr.py --dataset=dsprites --data_root=<PATH-TO-DATASET> --cyclic_trans --fixed_in_intervention=0,1,2,3 --fixed_in_sampling=0,2,3 --fixed_values=0,5,14 --distrib=uniform --displacement_range=-10,10 --n_steps=2 --rotate_actions=45 --num_train=50000 --batch_size=500 --epochs=501 --lr=0.0001 --toggle_training_every=6,4 --shuffle=1 --use_adam --use_cuda --conv_channels=32,32,32,32 --kernel_sizes=6,4,4,4 --strides=2,2,1,1 --lin_channels=1024 --net_act=relu --n_free_units=1 --dims=2,2 --group_hidden_units=64,64 --reconstruct_first --exponential_map --latent_loss --latent_loss_weight=400 --val_epoch=10 --num_val=500 --plot_epoch=20 --plot_manifold_latent=[0,1] --plot_pca --plot_vary_latents=[1,4,5]