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
# Base class for Flex 'FlexUnit' tests
#
from src import action
from src.globals import Define
from src import util

from src.mod.flex_app import FlexAppComponent


# ATLAS config interface
def FlexTest(name, **kwargs):
  Define(name, FlexTestComponent, **kwargs)


class FlexTestComponent(FlexAppComponent):

  def Init(self, **kwargs):
		FlexAppComponent.Init(self, **kwargs)

  def Test(self):
    # Requires 'flexor' test runner installation.
    # See http://code.google.com/p/flexor-test-runner
    cmd = action.TestCommand('flexor', desc=self.GetName())
    cmd.Add(util.AbsPath(self.target))
    return cmd.Run()
