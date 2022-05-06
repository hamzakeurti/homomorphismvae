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
# @title          :displacementae/grouprepr/representation_utils.py
# @author         :Hamza Keurti
# @contact        :hkeurti@ethz.ch
# @created        :24/03/2022
# @version        :1.0
# @python_version :3.7.4

from enum import Enum


class Representation(Enum):
    BLOCK_ROTS = 1
    MLP = 2
    BLOCK_MLP = 3
    PROD_ROTS_LOOKUP = 4
    LOOKUP = 5
    BLOCK_LOOKUP = 6
    UNSTRUCTURED = 7
