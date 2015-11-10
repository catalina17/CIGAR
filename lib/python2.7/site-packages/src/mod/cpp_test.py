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
# test_cpp.py -  Implements cpp_test component.
#
from src.globals import Define

from src.mod.cpp_exe import CppExeComponent
from src.mod.test import TestComponent


# ATLAS config interface
def CppTest(name, **kwargs):
  Define(name, CppTestComponent, **kwargs)


class CppTestComponent(CppExeComponent, TestComponent):
  """Cpp Unit Test Dependency."""

  def Init(self, **kwargs):
    CppExeComponent.Init(self, **kwargs)
    TestComponent.Init(self, self.target, **kwargs)
