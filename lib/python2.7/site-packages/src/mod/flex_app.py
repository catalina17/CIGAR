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
# Base class for Flex MX applications.
#
import re

from src import action
from src import filecache
from src.globals import Define, ConvertPaths
from src.registry import registry
from src import util

from src.mod.flex_binary import FlexBinaryComponent


# ATLAS config interface
def FlexApp(name, **kwargs):
  Define(name, FlexAppComponent, **kwargs)


class FlexAppComponent(FlexBinaryComponent):

  def Init(self, **kwargs):
    FlexBinaryComponent.Init(self, **kwargs)

    self.target = util.BuildPath(self.GetPath()) + '.swf'

  def Build(self):
    util.MakeParentDir(self.GetBuildDirectory())
    filecache.Purge(self.target)

    cmd = action.BuildCommand('mxmlc', 'flex build %s' % self.GetName())

    cmd.AddParamPath('-o', self.target)

    for lib in self.libs:
      cmd.Add('-compiler.library-path+=%s' % lib.target)

    for ext in self.ext_libs:
      cmd.Add('-compiler.library-path+=%s' % ext.GetPath())

    for s in self.sources:
      cmd.AddPath(s.GetPath())

    result = cmd.Run()
    # Strip output on success
    # TODO: Move to a Flex Binary base class
    if result.success:
      if (re.search('^Loading configuration file', result.output[0]) and
          re.search('\.swf \(\d+ bytes\)$', result.output[1])):
        result.output = []

    return result
