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
# Implements cpp_exe component. Defines a C++ executable program and its dependencies.
#
from src import action
from src import filecache
from src.flags import flags
from src.globals import Define
from src import util

from src.mod.cpp_binary import CppBinaryComponent


# ATLAS config interface
def CppExe(name, **kwargs):
  Define(name, CppExeComponent, **kwargs)

class CppExeComponent(CppBinaryComponent):
  def Init(self, ld_flags=None, **kwargs):
    target = util.BuildPath(self.GetPath())
    CppBinaryComponent.Init(self, target=target, **kwargs)
    self.ld_flags = ld_flags

  def AddLibrary(self, comp):
    if not comp in self.libs:
      # Unlike libraries, EXE's require their libraries to be built first.
      self.AddDepend(comp)
      self.libs.add(comp)

  def Build(self):
    util.MakeParentDir(self.target)
    filecache.Purge(self.target)

    cmd = action.BuildCommand('g++', 'link %s' % self.GetName())
    if flags.debug:
      cmd.Add('-g')
    cmd.AddFlags(self.ld_flags)
    cmd.AddParamPath('-o', self.target)

    for o in self.objects:
      cmd.AddPath(o.GetTargetName())

    for lib in self.OrderedLibraries():
      cmd.AddPath(lib.GetTargetName())

    for lib in self.ext_libs:
      cmd.Add('-l' + lib)

    return cmd.Run()

  def Run(self, args):
    # TODO: This functionality is shared with 'TestComponent'
    cmd = action.RunCommand(self.target)
    cmd.AddFlags(args)
    cmd.Run()

  def OrderedLibraries(self):
    """Build up a list of ordered dependencies, with reliant libs before dependants, as expected by g++ compiler."""
    # TODO: This is sort inefficient (especially last line, whcih runs through ALL libs many times, the last time being the entire list for the CppExe.libs list)
    # ... But it works :)
    # Also, don't want to import CppLibraryComponent!!!!!!!!!!
    from cpp_library import CppLibraryComponent
    ordered = []
    def AddLibraries(comp):
      for dep in comp.libs:
        if isinstance(dep, CppLibraryComponent):
          AddLibraries(dep)
      for dep in comp.libs:
        if dep not in ordered:
          ordered.append(dep)

    AddLibraries(self)
    ordered.reverse()
    return ordered
