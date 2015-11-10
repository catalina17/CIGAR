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
import re
import os

from src import action
from src.flags import flags
from src import filecache
from src import log
from src import util

from src.mod.cpp_include import CppIncludeComponent


CPP_EXT_RE = re.compile('\.(cpp|cc)$')


class CppSourceComponent(CppIncludeComponent):
  """Represents a .cpp dependency explicicly listed by a ATLAS muodule."""

  def Init(self, cc_flags=None, include_dirs=None, make_depends=True):
    CppIncludeComponent.Init(self, include_dirs, make_depends)
    self.target = util.BuildPath(CPP_EXT_RE.sub('.o', self.GetPath()))
    self.cc_flags = cc_flags or []

  def GetTargetName(self):
    return self.target

  def GetModifyTime(self):
    return util.GetModifyTime(self.target)

  def NeedsBuild(self, timestamp):
    if not util.Exists(self.target):
      return True
    source_time = util.GetModifyTime(self.source)
    return self.GetModifyTime() < max(source_time, timestamp)

  def Build(self):
    util.MakeParentDir(self.target)
    filecache.Purge(self.target)

    cmd = action.BuildCommand('g++', 'compile %s' % self.GetName())
    cmd.Add('-Wall')
    cmd.Add('-c')
    if flags.debug:
      cmd.Add('-g')
    cmd.AddFlags(self.cc_flags)

    cmd.AddParamPath('-o', self.target)
    cmd.AddPath(self.source)

    for dir in self.include_dirs:
      cmd.AddParamPath('-I', dir)
    return cmd.Run()

  def Clean(self):
    return action.DeleteCommand(self.target).Run()
