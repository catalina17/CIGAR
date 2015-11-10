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
# shell_builder.py - Use shell scripts to build targets.
#
from src import action
from src.component import Component
from src import filecache
from src.globals import Define
from src.registry import registry
from src import util

from src.mod.file import FileComponent


# ATLAS config interface
def ShellBuilder(name, **kwargs):
  Define(name, ShellBuilderComponent, **kwargs)


class ShellBuilderComponent(Component):

  def Init(self, input, output, **kwargs):
    Component.Init(self, **kwargs)

    self.script = util.NameToPath(self.GetName())

    self.input = []
    for i in self.ConvertPaths(input):
      comp = registry.Define(i, FileComponent)
      self.input.append(comp)
      self.AddDepend(comp)

    self.targets = []
    for s in self.ConvertPaths(output):
      self.targets.append(util.BuildPath(util.NameToPath(s)))

  def GetModifyTime(self):
    return max([util.GetModifyTime(t) for t in self.targets])

  def NeedsBuild(self, timestamp):
    for t in self.targets:
      if not util.Exists(t):
        return True
    for target in self.targets:
      filecache.Purge(target)

    script_time = util.GetModifyTime(self.script)
    return self.GetModifyTime() < max(script_time, timestamp)

  def Build(self):
    for t in self.targets:
      util.MakeParentDir(t)
      filecache.Purge(t)

    util.SetExecutePermission(self.script)
    cmd = action.BuildCommand(self.script, 'shell script %s' % self.GetName())
    return cmd.Run()

  def Clean(self):
    return [action.DeleteCommand(t).Run() for t in self.targets]
