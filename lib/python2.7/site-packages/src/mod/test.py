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
# test.py - base test coponent
#
from src import action
from src.component import Component
from src import util


class TestComponent(Component):

  def Init(self, target, **kwargs):
    Component.Init(self, **kwargs)
    self.test_target = target

  def Test(self):
    util.SetExecutePermission(self.test_target)
    path = util.AbsPath(self.test_target)  # CWD is set to temp dir, so we must use full path
    return action.TestCommand(action=path, desc=self.GetName()).Run()
