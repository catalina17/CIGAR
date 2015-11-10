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
# automake.py
# Desc: Provides wrapper for including 'Automake' projects. Automake is a common
#       build system used to distribute many open source projects.
#
import os

from src import action
from src.component import Component
from src.extension import Extension
from src import filecache
from src.flags import flags
from src.globals import Define, ConvertPaths
from src.registry import registry
from src import util

from src.mod.automake_config import AutomakeConfigComponent
from src.mod.cpp_binary import CppBinaryComponent
from src.mod.file import FileComponent

# Common location within build tree where all Automake projects are installed.
AUTOMAKE_INSTALL_DIR = 'AUTOMAKE_INSTALL'


# ATLAS config interface
def Automake(name, **kwargs):
  Define(name, AutomakeComponent, **kwargs)


class AutomakeLibraryExtension(Extension):
  def Depend(self, depend, reliant):
    reliant.AddDepend(depend)
    reliant.AddSourceDepend(depend)
    for dir in depend.include_dirs:
      reliant.AddIncludeDirectory(dir)
    for lib in depend.library_targets:
      reliant.AddLibrary(lib)
    for lib in depend.ext_libs:
      reliant.AddExternalLibrary(lib)


def Init():
  registry.RegisterExtension('automake_library',
                             CppBinaryComponent,
                             AutomakeLibraryExtension(AutomakeComponent,
                                                      convert_paths=True))


class AutomakeComponent(Component):

  def Init(self, library_target=None, configure_flags=None, include_dir=None, ext_library=None, **kwargs):
    Component.Init(self, **kwargs)

    self.ext_libs = set()  # Just set of strings for now...

    self.AddDepend(registry.Define(util.ConvertPaths('configure', self.GetDirectory()),
                                   AutomakeConfigComponent,
                                   configure_flags=configure_flags,
                                   install_dir=self.GetInstallDirectory()))

    self.library_targets = set()
    for lib in (library_target or []):
      if not lib.endswith('.a'): lib += '.a'
      lib = util.ConvertPaths(lib, self.GetLibraryInstallDirectory())
      comp = registry.Define(lib, FileComponent)
      self.library_targets.add(comp)
      comp.OnInit()  # Need to either force initialization or else call self.AddDepend(comp)

    self.include_dirs = [self.GetIncludeInstallDirectory()]
    for dir in util.MakeArgumentList(include_dir):
      dir = os.path.join(self.GetIncludeInstallDirectory(), dir)
      self.include_dirs.append(dir)

    for lib in util.MakeArgumentList(ext_library):
      self.ext_libs.add(lib)

  def GetInstallDirectory(self):
    return util.BuildPath(AUTOMAKE_INSTALL_DIR)

  def GetIncludeInstallDirectory(self):
    return os.path.join(self.GetInstallDirectory(), 'include')

  def GetLibraryInstallDirectory(self):
    return os.path.join(self.GetInstallDirectory(), 'lib')

  def GetModifyTime(self):
    return max([t.GetModifyTime() for t in self.library_targets])

  def NeedsBuild(self, timestamp):

    if timestamp > self.GetModifyTime():
      return True

    # THis is a very fast check that avoids calling the makefile on every infocation
    # (as below). The disadvantage is that it makes very risky assumptions that
    # the package only needs to be rebuilt if the ./configure targets are modified.
    return False


    # It should be rare for automake projects to require a build.
    # So we need to make it very fast to return 'false'.
    # We'll try running make in 'question' mode, but that might not be fast engouh.
    # Also , by checking here, we prevent the standard 'Clean' command from
    # being called before 'Build', which forces a rebuild every time.
    cmd = action.BuildCommand(action='make',
                              work_dir=self.GetBuildDirectory())
    cmd.Add('-q')  # 'Question mode': Returns 0 if no build needed, 1 otherwise.
    result = cmd.Run();
    return not result.success

  def Build(self):
    util.MakeDir(self.GetBuildDirectory())
    for t in self.library_targets:
      filecache.Purge(t.GetPath())

    # This 'installs' the libraries to our build directory (specified
    # by AutomakeConfig's build step.
    cmd = action.BuildCommand(action='make install',
                              desc='automake %s' % self.GetName(),
                              work_dir=self.GetBuildDirectory())
    result = cmd.Run();

    # if an automake project succeeds, we don't care about the output
    if result.success and not flags.verbose:
      result.output = []

    return result

  def Clean(self):
    # TODO: Do we want this to run very often?

    if os.path.isdir(self.GetBuildDirectory()):
      return action.BuildCommand('make uninstall && make clean',
                                 desc='uninstalled %s' % self.GetName(),
                                 work_dir=self.GetBuildDirectory()).Run()
    else:
      return None
