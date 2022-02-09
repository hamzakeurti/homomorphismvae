#!/usr/bin/env python3
# Copyright 2022 Hamza Keurti
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
# @title          :displacementae/autoencoder/train_prodred.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :09/02/2022
# @version        :1.0
# @python_version :3.7.4

import __init__

import autoencoder.train_utils as tutils

if __name__=='__main__':
    tutils.run(mode='prodrep')