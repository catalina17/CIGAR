#!/usr/bin/python2.5
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
import os
import stat
import unittest

import constants
import filecache
import metrics


class FileCacheTest(unittest.TestCase):

  def test_cycle(self):
    path = '.filecache_test_cycle___delete_me'

    if os.path.exists(path):
      os.remove(path)

    self.assertFalse(filecache.Exists(path))
    self.assertEquals(constants.MAX_AGE, filecache.GetModifyTime(path))

    open(path, 'w').write('hi')
    filecache.Purge(path)
    self.assertTrue(os.path.exists(path))
    self.assertTrue(filecache.Exists(path))
    self.assertEquals(os.stat(path)[stat.ST_MTIME], filecache.GetModifyTime(path))

    filecache.Delete(path)
    self.assertFalse(os.path.exists(path))
    self.assertFalse(filecache.Exists(path))
    self.assertEquals(constants.MAX_AGE, filecache.GetModifyTime(path))


if __name__ == '__main__':
  unittest.main() 
