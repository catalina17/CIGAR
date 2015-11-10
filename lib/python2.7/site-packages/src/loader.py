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
# Process ATLAS configuration files
#
import os

import constants
from globals import glob
import metrics
from registry import registry
import util

from mod.file import FileComponent

# Configuration File Directives
from mod.automake import Automake
from mod.cpp_exe import CppExe
from mod.cpp_library import CppLibrary
from mod.cpp_test import CppTest
from mod.flex_app import FlexApp
from mod.flex_library import FlexLibrary
from mod.flex_test import FlexTest
from mod.protocol_buffer import ProtocolBuffer
from mod.python_test import PythonTest
from mod.shell_builder import ShellBuilder
from mod.shell_test import ShellTest


# TODO: TEMP - clean me up!
from mod import automake
from mod import cpp_library
from mod import flex_library
from mod import protocol_buffer


class Loader(object):

  def __init__(self, deps):
    self.path_tree = []
    self.current_build_file = None
    self.graph = deps

    # TODO: Load modules dynamically
    automake.Init()  # TODO: Need better location
    cpp_library.Init()  # TODO: Need better location
    flex_library.Init()  # TODO: Need better location
    protocol_buffer.Init()  # TODO: Need better location

  def ConvertPaths(self, paths):
    return util.ConvertPaths(paths, self.GetCurrentDirectory())

  def GetCurrentDirectory(self):
    return self.current_directory

  def Parse(self, path):
    exec(open(path))
    metrics.config_files += 1

  def Load(self, dir):
    for name in os.listdir(dir):
      path = util.RelPath(os.path.join(dir, name))
      if os.path.isdir(path) and path != constants.BUILD_ROOT:
          self.Load(path)  # Recursive

          file = os.path.join(path, constants.BUILD_SOURCE_FILE)
          if os.path.isfile(file):
            self.current_directory = path

            registry.RegisterModule(path)
            util.MakeDir(util.BuildPath(path))

            name = util.ConvertPaths(constants.BUILD_SOURCE_FILE, path)
            comp = registry.Define(name, FileComponent)
            self.current_build_file = comp

            self.Parse(comp.GetPath())

  def Define(self, path, comp_type, **kwargs):
    comp = registry.Define(path, comp_type, **kwargs)
    # All build configs depend on their source ATLAS file
    if self.current_build_file:
      comp.AddDepend(self.current_build_file)
    return comp

