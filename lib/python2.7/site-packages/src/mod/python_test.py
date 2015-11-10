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
# python_test.py - Python test support.
#
from src import action
from src.globals import Define

from src.mod.file import FileComponent
from src.mod.test import TestComponent


# ATLAS config interface
def PythonTest(name, **kwargs):
  Define(name, PythonTestComponent, **kwargs)


class PythonTestComponent(FileComponent, TestComponent):
  """Python PyUnit Test Dependency."""

  def Init(self, **kwargs):
    FileComponent.Init(self, **kwargs)
    TestComponent.Init(self, self.source, **kwargs)
