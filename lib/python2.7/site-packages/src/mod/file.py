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


from src.component import Component
from src import util


class FileComponent(Component):
  """Dependency on a single file."""
  def Init(self, source=None, **kwargs):
    Component.Init(self, **kwargs)
    self.source = source
    if not source:
      self.source = self.GetPath()

  def GetModifyTime(self):
    return util.GetModifyTime(self.source)

  def NeedsBuild(self, timestamp):
    return False  # Never need to re-generate

  def IsDefined(self):
    return True  # Raw files don't need to an explict gen rule

  # TODO: Do derived classes shadow this?
  def GetTargetName(self):
    return self.source
