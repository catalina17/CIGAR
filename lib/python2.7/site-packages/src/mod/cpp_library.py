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
# Implements cpp_lib component.
#
import os

from src import action
from src.extension import Extension
from src import filecache
from src.globals import Define
from src.registry import registry
from src import util

from src.mod.cpp_binary import CppBinaryComponent


# ATLAS config interface
def CppLibrary(name, **kwargs):
  Define(name, CppLibraryComponent, **kwargs)


class LibExtension(Extension):
  def Depend(self, depend, reliant):
    if not depend.EmptyTarget():
      reliant.AddLibrary(depend)
    for lib in depend.libs:
      reliant.AddLibrary(lib)
    for lib in depend.ext_libs:
      reliant.AddExternalLibrary(lib)
    for dir in depend.include_dirs:
      reliant.AddIncludeDirectory(dir)
    for sd in depend.source_depends:
      reliant.AddSourceDepend(sd)


def Init():
  registry.RegisterExtension('library',
                             CppBinaryComponent,
                             LibExtension(CppLibraryComponent,
                                          convert_paths=True))


class CppLibraryComponent(CppBinaryComponent):

  def Init(self, **kwargs):
    target = util.BuildPath(os.path.join(self.GetDirectory(),
                                         'lib%s.a' % self.GetShortName()))
    CppBinaryComponent.Init(self, target=target, **kwargs)

  def AddLibrary(self, comp):
    # Dependant libraries don't need to be built before reliant libraries. This greatly improves parallelism (takes 3/4 as long in my tests).
    self.libs.add(comp)

  def EmptyTarget(self):
    """Its possible to have a CPP library target that has no source (more of a
       dependency bundle)."""
    return not self.objects

  def NeedsBuild(self, timestamp):
    if self.EmptyTarget():
      return False
    return CppBinaryComponent.NeedsBuild(self, timestamp)

  def Build(self):
    util.MakeParentDir(self.target)
    filecache.Purge(self.target)

    cmd = action.BuildCommand('ar', 'build %s' % self.GetName())
    cmd.Add('rcs')
    cmd.AddPath(self.target)

    for o in sorted(self.objects, util.CompareName):
      cmd.AddPath(o.GetTargetName())

    return cmd.Run()
