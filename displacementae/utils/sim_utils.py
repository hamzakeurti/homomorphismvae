#!/usr/bin/env python3
# Copyright 2021 Hamza Keurti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @title          :displacementae/utils/sim_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :16/11/2021
# @version        :1.0
# @python_version :3.7.4

import os
import select
import sys
from warnings import warn
import logging
import shutil
import torch
import numpy as np
import random
import pickle
import json

from displacementae.utils import logger_config


BCE_LOWEST = 'bce_lowest'
KL_HIGHEST = 'kl_highest'
LOSS_LOWEST = 'loss_lowest'
LOSS_LOWEST_EPOCH = 'loss_lowest_epoch'
BCE_FINAL = 'bce_final'
KL_FINAL = 'kl_final'
LOSS_FINAL = 'loss_final'
FINISHED = 'finished'
NUM_WEIGHTS = 'num_weights'

_SUMMARY_KEYWORDS = [
    # The weird prefix "aa_" makes sure keywords appear first in the result csv.
    LOSS_FINAL,
    LOSS_LOWEST,
    LOSS_LOWEST_EPOCH,

    BCE_FINAL,
    BCE_LOWEST,
    
    # The following are only relevant for variational models.
    KL_FINAL,
    KL_HIGHEST,
    
    
    NUM_WEIGHTS,

    # Should be set in your program when the execution finished successfully.
    FINISHED
]


def setup_environment(config):
    """
    Sets up output directory and logger.
    """
    ### Output folder.
    if os.path.exists(config.out_dir):
        # TODO allow continuing from an old checkpoint.
        print('The output folder %s already exists. ' % (config.out_dir) + \
              'Do you want us to delete it? [y/n]')
        inps, _, _ = select.select([sys.stdin], [], [], 30)
        if len(inps) == 0:
            warn('Timeout occurred. No user input received!')
            response = 'n'
        else:
            response = sys.stdin.readline().strip()
        if response != 'y':
            raise IOError('Could not delete output folder!')
        shutil.rmtree(config.out_dir)

        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    else:
        os.makedirs(config.out_dir)
        print("Created output folder %s." % (config.out_dir))

    figs_dir = os.path.join(config.out_dir,'figures')
    if not config.no_plots and not os.path.exists(figs_dir):
        os.makedirs(figs_dir)
    
    ckpt_dir = os.path.join(config.out_dir,'checkpoint')
    if config.checkpoint and not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir) 

    # Save user configs to ensure reproducibility of this experiment.
    with open(os.path.join(config.out_dir, 'config.pickle'), 'wb') as f:
        pickle.dump(config, f)
    # A JSON file is easier to read for a human.
    with open(os.path.join(config.out_dir, 'config.json'), 'w') as f:
        json.dump(vars(config), f,indent=4,sort_keys=True)
    
    ### Logger
    logger_name = 'logger'
    logger = logger_config.config_logger(logger_name,
            os.path.join(config.out_dir, 'logfile.txt'),
            logging.DEBUG, logging.INFO)
    
    ### Deterministic computation.
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed_all(config.random_seed)
    np.random.seed(config.random_seed)
    random.seed(config.random_seed)

    ### Torch Device
    use_cuda = config.use_cuda
    if use_cuda and config.cuda_number is not None:
        cuda_number = config.cuda_number
        device = torch.device(f"cuda:{cuda_number}")
        logger.info(f'Using cuda : {use_cuda} -- Device Number {cuda_number}')
    else:
        device = torch.device("cuda" if use_cuda else "cpu")
        logger.info(f'Using cuda : {use_cuda}')
    
    return device, logger

def setup_summary_dict(config, shared, nets):
    """Setup the summary dictionary that is written to the performance
    summary file (in the result folder).

    This method adds the keyword "summary" to `shared`.

    Args:
        config: Command-line arguments.
        shared: Miscellaneous data shared among training functions (summary dict
            will be added to this :class:`argparse.Namespace`).
    """
    summary = dict()

    num = sum([p.numel() for p in nets.parameters() if p.requires_grad])


    # Note, we assume that all configs have the exact same keywords.
    summary_keys = _SUMMARY_KEYWORDS

    for k in summary_keys:
        if k in [LOSS_FINAL,BCE_FINAL,KL_FINAL,LOSS_LOWEST,KL_HIGHEST,
                 BCE_LOWEST,LOSS_LOWEST_EPOCH]:
            summary[k] = -1
        elif k == NUM_WEIGHTS:
            summary[k] = num
        elif k == 'finished':
            summary[k] = 0
        else:
            # Implementation must have changed if this exception is
            # raised.
            raise ValueError('Summary argument %s unknown!' % k)

    shared.summary = summary


def save_summary_dict(config, shared):
    """Write a text file in the result folder that gives a quick
    overview over the results achieved so far.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    # "setup_summary_dict" must be called first.
    assert(hasattr(shared, 'summary'))

    summary_fn = 'performance_overview.txt'

    with open(os.path.join(config.out_dir, summary_fn), 'w') as f:
        for k, v in shared.summary.items():
            if isinstance(v, float):
                f.write('%s %f\n' % (k, v))
            else:
                f.write('%s %d\n' % (k, v))

def backup_cli_command(config):
    """Write the curret CLI call into a script.

    This will make it very easy to reproduce a run, by just copying the call
    from the script in the output folder. However, this call might be ambiguous
    in case default values have changed. In contrast, all default values are
    backed up in the file ``config.json``.

    Args:
        (....): See docstring of function :func:`setup_summary_dict`.
    """
    script_name = sys.argv[0]
    run_args = sys.argv[1:]
    command = 'python3 ' + script_name
    # FIXME Call reconstruction fails if user passed strings with white spaces.
    for arg in run_args:
        command += ' ' + arg

    fn_script = os.path.join(config.out_dir, 'cli_call.sh')

    with open(fn_script, 'w') as f:
        f.write('#!/bin/sh\n')
        f.write('# The user invoked CLI call that caused the creation of\n')
        f.write('# this output folder.\n')
        f.write(command)

def save_dictionary(shared,config):
    with open(os.path.join(config.out_dir, 'stats.json'),'w') as f:
        json.dump(vars(shared),f,indent=4)
