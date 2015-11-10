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
import metrics
import util


# The master loader instance. A (supposedly) necessary evil due to the ATLAS interface below
master_loader = None

def Define(name, comp_type, **kwargs):
  metrics.total_modules += 1
  master_loader.Define(ConvertPaths(name), comp_type, **kwargs)

def ConvertPaths(paths):
  return master_loader.ConvertPaths(paths)


def glob(path):
  return util.GlobPath(path, master_loader.GetCurrentDirectory())


