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
# Base class for Flex binary targets (swf/swc)
#
#  flex_binary.py
#
from src import action
from src.component import Component
from src.registry import registry
from src import util

from src.mod.file  import FileComponent


class FlexBinaryComponent(Component):

  def Init(self, source=None, external_library=None, data=None, **kwargs):
    Component.Init(self, **kwargs)

    self.libs = set()

    self.sources = set()
    for s in self.ConvertPaths(source):
      self.AddSource(registry.Define(s, FileComponent))

    self.ext_libs = set()
    for ext in self.ConvertPaths(external_library):
      self.AddExternalLibrary(registry.Define(ext, FileComponent))

    for d in self.ConvertPaths(data):
      self.AddDepend(registry.Define(d, FileComponent))

    self.params = set()  # TEMP workaround for core PB lib

    self.ProcessExtensions(kwargs)  # TODO: Move to base class

  def AddParam(self, param):
    self.params.add(param)

  def AddSource(self, comp):
    self.sources.add(comp)
    self.AddDepend(comp)

  def AddLibrary(self, comp):
    self.libs.add(comp)
    self.AddDepend(comp)

  def AddExternalLibrary(self, ext_lib):
    self.ext_libs.add(ext_lib)

  def GetModifyTime(self):
     return util.GetModifyTime(self.target)

  def NeedsBuild(self, timestamp):
    if not util.Exists(self.target):
      return True
    return timestamp > self.GetModifyTime()

  def Clean(self):
    return action.DeleteCommand(self.target).Run()
