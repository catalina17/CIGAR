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
# Protocol Buffer Dependency
#
import os
import re
from string import capitalize;

from src import action
from src.extension import Extension
from src import filecache
from src.globals import Define, ConvertPaths
from src.registry import registry
from src import util

from src.mod.cpp_binary import CppBinaryComponent
from src.mod.cpp_include import CppIncludeComponent
from src.mod.cpp_source import CppSourceComponent
from src.mod.flex_binary import FlexBinaryComponent
from src.mod.flex_library import FlexLibraryComponent
from src.mod.file import FileComponent


# Protocol Buffer file extension
PROTO_EXT_RE = re.compile('\.proto$')

# Matches protocol buffer import statements (within .proto files)
PROTO_IMPORT_RE = re.compile('^\s*import\s+"([^"]+)"\s*;', re.M)


class CppBinaryExtension(Extension):
  def Depend(self, depend, reliant):
    reliant.AddExternalLibrary('protobuf')
    reliant.AddIncludeDirectory(depend.GetBuildDirectory())
    reliant.AddLibrary(depend.targets['cpp'])
    # Simple way to get all dependant *.o files
    for imported in depend.imported:
      imported.OnInit()  # TODO: Need better way to call Init()
      self.Depend(imported, reliant)


class FlexBinaryExtension(Extension):
  def Depend(self, depend, reliant):
    reliant.AddSource(depend.targets['as'])
    # TODO(mparent): CPP & FLex shouldn't rely on hard-coded libs?
    reliant.AddParam('-compiler.library-path+=build/debug/third_party/protocol-buffers-actionscript/protobuf.swc')
    #for dep in self.depends:
    #      self.targets['as'].AddDepend(dep)

    # Simple way to get all dependant *.as files
    for imported in depend.imported:
      imported.OnInit()  # TODO: Need better way to call Init()
      self.Depend(imported, reliant)


def Init():
  registry.RegisterExtension('protobuf',
                             CppBinaryComponent,
                             CppBinaryExtension(ProtocolBufferComponent,
                                                convert_paths=True))
  registry.RegisterExtension('protobuf',
                             FlexBinaryComponent,
                             FlexBinaryExtension(ProtocolBufferComponent,
                                                 convert_paths=True))


# ATLAS config interface
def ProtocolBuffer(name, **kwargs):
  Define(name, ProtocolBufferComponent, **kwargs)


class ProtocolBufferComponent(FileComponent):

  def Init(self, **kwargs):
    FileComponent.Init(self)

    def FlexTargetName(ext):
      """Flex requires CamelCase.as"""
      name = PROTO_EXT_RE.sub(ext, self.GetShortName())
      name = "".join([capitalize(w) for w in name.split('_')])
      return util.PathToName(os.path.join(self.GetBuildDirectory(), name))

    def TargetName(ext):
      return util.PathToName(util.BuildPath(PROTO_EXT_RE.sub(ext, self.GetPath())))

    dirs = [util.GetBuildDirectory()]
#as_target = util.PathToName(PROTO_EXT_RE.sub('.as', self.GetPath
    self.targets = {
                    'as': registry.Define(FlexTargetName('.as'),
                                          FileComponent),
                    'cpp': registry.Define(TargetName('.pb.cc'),
                                           CppSourceComponent,
                                           include_dirs=dirs,
                                           make_depends=False),  #PB import dependency handled by ProtocolBufferComponent
                    'h': registry.Define(TargetName('.pb.h'),
                                         CppIncludeComponent,
                                         make_depends=False),
                    'py': registry.Define(TargetName('_pb2.py'),
                                          FileComponent),
                   }

    for target in self.targets.values():
      target.AddDepend(self)

    # Directories are searched in order
    # TODO: Support relative directories. Mixing absolute/relative directories seems to confuse protoc compiler.
    self.import_dirs = [util.GetRootDirectory(),
                        #self.GetDirectory(),
                       ]
    # Add import dependencies
    self.imported = []
    self.AddImportDepends()

  def AddImportDepends(self):
    """Read file and look for any .proto import dependencies."""
    for path in util.GetFileMatches(self.source, PROTO_IMPORT_RE, 'Protocol Buffer import'):
      count = 0
      for dir in self.import_dirs:
        imported = util.RelPath(os.path.join(dir, path))
        if '/' in imported:
          pieces = imported.rpartition('/')
          imported = pieces[0] + ':' + pieces[2]
          comp = registry.Get(imported)
          if comp:
            self.AddDepend(comp)
            self.imported.append(comp)
            count += 1
      if count == 0:
        raise util.ConfigError("Protocol Buffer file %s imports '%s', which is not defined in any ATLAS configuration file. Did you forget to add a ProtocolBuffer definition for this file?" % (self, imported))
      elif count > 1:
        raise util.ConfigError("Protocol Buffer file %s imports '%s', which has two possible matches (relative and absolute paths)")

  def GetBuildDirectory(self):
    return util.BuildPath(self.GetDirectory())

  def GetCppTarget(self):
    return util.BuildPath(PROTO_EXT_RE.sub('.pb.cc', self.GetPath()))  #TODO: Redundant
    # MJP: I think this is cleaner (but haven't tested):
    #return self.targets['cpp'].GetPath()

  def CreatePythonTrail(self):
    """Create python 'init' files to allow nested module dependencies."""
    dir = self.GetBuildDirectory()
    while len(dir) >= len(util.BuildPath()):
      path = os.path.join(dir, '__init__.py')
      open(path, 'w').close()  # Empty file
      dir = os.path.dirname(dir)

  def GetModifyTime(self):
    return max([util.GetModifyTime(t.GetPath()) for t in self.targets.values()])

  def NeedsBuild(self, timestamp):
    for target in self.targets.values():
      if not util.Exists(target.GetPath()):
        return True
    source_time = util.GetModifyTime(self.source)
    return self.GetModifyTime() < max(source_time, timestamp)

  def Build(self):
    util.MakeDir(self.GetBuildDirectory())
    for target in self.targets.values():
      filecache.Purge(target.GetPath())

    self.CreatePythonTrail()

    cmd = action.BuildCommand('protoc', 'proto-compile %s' % self.GetName())  # Protocol Buffer Compiler
    cmd.Add('--as3_out=%s' % util.BuildPath())
    cmd.Add('--cpp_out=%s' % util.BuildPath())
    cmd.Add('--python_out=%s' % util.BuildPath())
    cmd.Extend('--proto_path=%s' % util.RelPath(dir) for dir in self.import_dirs)

    cmd.AddPath(self.source)
    return cmd.Run()

  def Clean(self):
    results = []
    for target in self.targets.values():
      r = action.DeleteCommand(target.GetPath()).Run()
      results.append(r)
    return results
