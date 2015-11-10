#!/usr/bin/python
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
#
import unittest
import util
import os

from flags import flags


class UtilTest(unittest.TestCase):

  def test_name_to_path(self):
    self.assertEquals('a', util.NameToPath('a'))
    self.assertEquals('a/b', util.NameToPath('a:b'))
    self.assertEquals('a/b/c', util.NameToPath('a/b:c'))

  def test_rel_path(self):
    cwd = os.getcwd()
    self.assertEquals('.', util.RelPath(cwd))
    self.assertEquals('a', util.RelPath(os.path.join(cwd, 'a')))
    self.assertEquals('a', util.RelPath('./a'))

  def test_abs_path(self):
    self.assertEquals(os.path.abspath('out'), util.AbsPath('out'))

  def test_strip_build_dir(self):
    self.assertEquals('a/b/c', util.StripBuildDirectory('build/opt/a/b/c'))
    self.assertEquals('a/b/c', util.StripBuildDirectory('a/b/c'))

  def test_build_path(self):
    # Relative
    flags.debug = True
    self.assertEquals('build/debug/test', util.BuildPath('test'))
    self.assertEquals('build/debug/a/b', util.BuildPath('a/b'))

    # Absolute
    cwd = os.getcwd()
    path = os.path.join(cwd, 'a/b')
    build_path = os.path.join(cwd, 'build/debug/a/b')
    self.assertEquals(build_path, util.BuildPath(path))

    # Opt dir
    flags.debug = False
    self.assertEquals('build/opt/a/b', util.BuildPath('a/b'))

  def test_convert_path(self):
    self.assertEquals('a', util.ConvertPaths('a', None))
    self.assertEquals('a', util.ConvertPaths('a', ''))

    self.assertEquals('a:b', util.ConvertPaths('b', 'a'))
    self.assertEquals('a:b', util.ConvertPaths('a:b', None))

    self.assertEquals('a/b:c', util._ConvertPath('b:c', 'a'))
    self.assertEquals('a/b:c', util._ConvertPath('c', 'a/b'))
    self.assertEquals('a/b:c', util._ConvertPath('c', 'a/b'))

    self.assertEquals('a/b/c:d', util._ConvertPath('b/c:d', 'a'))

    self.assertEquals('a/b:c', util._ConvertPath('b/c', 'a'))

  def test_convert_paths(self):
    self.assertEquals('a:b', util.ConvertPaths('b', 'a'))
    self.assertEquals(['a:b', 'a:c'], util.ConvertPaths(['b', 'c'], 'a'))

  def testDescribeSystemSignal(self):
    self.assertEqual(util.DescribeSystemSignal(10, platform=util.PLATFORM_OSX),
                     ("SIGBUS", "Bus Error (bad memory access)"))
    self.assertEqual(util.DescribeSystemSignal(1000)[0],
                     'SIGNAL 1000')


if __name__ == "__main__":
  unittest.main()
