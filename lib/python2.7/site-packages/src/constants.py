#---------------------------------------------------------------------------
# Copyright 2008 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#---------------------------------------------------------------------------
#
# System-wide constants
#
import os
import sys


VERSION = '0.27.0'  # Atlas Version

# Scratch build directory
BUILD_ROOT = 'build'

DEBUG_BUILD_DIR = os.path.join(BUILD_ROOT, 'debug')
OPT_BUILD_DIR = os.path.join(BUILD_ROOT, 'opt')

# Filename of build configuration files.
BUILD_SOURCE_FILE = 'ATLAS'

# Used to compute # of workers in thread pool.
#   num_workers = X * num_processors
WORKERS_PER_PROCESSOR = 2

# TODO: Up these timeouts as needed
TEST_TIMEOUT = 60  # TODO: Base on test size
BUILD_TIMEOUT = (2 * 60) # TODO: Allow components to set timeout

# Max number of test stdout/stderr lines printed on failing tests.
MAX_TEST_OUTPUT_LINES = 200

# Time set by Build operations on rebuild
MIN_AGE = sys.maxint
MAX_AGE = 0
