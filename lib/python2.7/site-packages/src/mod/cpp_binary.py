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
# Base class for C++ EXE & LIB components
#
from src import action
from src.component import Component
from src.registry import registry
from src import util

from src.mod.cpp_source import CppSourceComponent


class CppBinaryComponent(Component):

  def Init(self, target, source=None, cc_flags=None, ld_flags=None, ext_library=None, **kwargs):
    Component.Init(self, **kwargs)
    self.target = target
    self.cc_flags = cc_flags
    self.include_dirs = set()
    self.ext_libs = set()  # Just set of strings for now...
    self.libs = set()
    self.objects = set()
    self.source_depends = set()  # Extra depends added to all sources

    self.AddIncludeDirectory(util.GetRootDirectory())
    self.AddIncludeDirectory(util.BuildPath(util.GetRootDirectory())) # generted files

    self.ProcessExtensions(kwargs)  # TODO: Move to base class

    for library in (ext_library or []):
      self.AddExternalLibrary(library)

    for s in self.ConvertPaths(source):
      comp = registry.Define(s,
                             CppSourceComponent,
                             cc_flags=self.cc_flags,
                             include_dirs=self.include_dirs)
      self.AddSource(comp)

  def AddIncludeDirectory(self, dir):
    assert isinstance(dir, str)
    dir = util.RelPath(dir)
    self.include_dirs.add(dir)

  def AddSourceDepend(self, comp):
    # Use case: All source files depend on an Automake project
    # being built first.
    self.source_depends.add(comp)

  def AddSource(self, comp):
    assert isinstance(comp, CppSourceComponent)
    self.AddDepend(comp)
    self.objects.add(comp)
    for depend in self.source_depends:
      comp.AddDepend(depend)

  def AddExternalLibrary(self, lib):
    assert isinstance(lib, str)
    self.ext_libs.add(lib)

  def GetModifyTime(self):
     return util.GetModifyTime(self.target)

  def GetTargetName(self):
    return self.target

  def NeedsBuild(self, timestamp):
    if not util.Exists(self.target):
      return True
    return timestamp > self.GetModifyTime()

  def Clean(self):
    return action.DeleteCommand(self.target).Run()
