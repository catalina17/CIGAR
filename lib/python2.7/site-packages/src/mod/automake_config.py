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
# automake_config.py
# Desc: Wrapper around automake's './confgure' build command.
#       The 'AutomakeComponent' depends on this component.
#
import os
import shutil

from src import action
from src.component import Component
from src import filecache
from src.flags import flags
from src import util


class AutomakeConfigComponent(Component):

  def Init(self, install_dir, configure_flags=None, **kwargs):
    Component.Init(self, **kwargs)

    self.install_dir = install_dir
    self.targets = []
    self.configure_flags = util.MakeArgumentList(configure_flags)
    # List of targets built by ./confgure script and required for Automake
    # TODO: Verify that this is a standard list, else make params.
    for target in ['Makefile', 'config.status']:
      self.targets.append(os.path.join(self.GetBuildDirectory(),
                                       target))

  def GetModifyTime(self):
    return max([util.GetModifyTime(t) for t in self.targets])

  def NeedsBuild(self, timestamp):
    for target in self.targets:
      if not util.Exists(target):
        return True

    return timestamp > self.GetModifyTime()

  def Build(self):
    for target in self.targets:
      filecache.Purge(target)

    shutil.copytree(self.GetDirectory(), self.GetBuildDirectory())

    cmd = action.BuildCommand(action='./configure', #os.path.join(util.AbsPath(self.GetDirectory()), 'configure'),
                              desc='configure %s' % self.GetName(),
                              work_dir=self.GetBuildDirectory())
    #cmd.Add('-C')  # Cache results -- not sure if this works/helps
    cmd.Add('--prefix="%s"' % util.AbsPath(self.install_dir))
    cmd.AddFlags(self.configure_flags)

    # TODO: Add optimization flags for -opt builds

    result = cmd.Run()
    # if an automake project succeeds, we don't care about the output
    if result.success and not flags.verbose:
      result.output = []

    return result

  def Clean(self):
    # TODO: Need better way for Clean() to run multiple commands

    if os.path.isdir(self.GetBuildDirectory()):
      # 'distclean' make target cleans everything, incluing ./configure output.
      result = action.BuildCommand('make distclean',
                                    desc='cleaned %s' % self.GetName(),
                                    work_dir=self.GetBuildDirectory()).Run()

      shutil.rmtree(self.GetBuildDirectory())

      return result
    else:
      return None
