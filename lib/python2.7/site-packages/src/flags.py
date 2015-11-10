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
# Command Line Flags
#


class Flags(object):
  def __init__(self):
    self.color = True       # if true, print colored output.
    self.debug = True       # build configuration: 'debug' or 'opt'.
    self.metrics = False    # if 'true', print extra metrics debug info.
    self.parallel = True    # if 'true', runs commands in parallel. else serially.
    self.verbose = False    # if 'true', print out extra descriptive info.


flags = Flags()  # Global instance
