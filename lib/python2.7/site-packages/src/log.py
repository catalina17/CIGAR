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
# Logging interface
#

from flags import flags

# Place holder for more sophisticated logging behavior

def Info(msg):
  if flags.verbose:
    print "INFO: ", msg


def Warning(msg):
  if flags.verbose:
    print "WARNING: ", msg


def Error(msg):
  print "ERROR: ", msg
