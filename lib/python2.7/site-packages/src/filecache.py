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
import os
import stat

import constants
import metrics


_cache = {}  # Store filesystem info to (drastically) reduce disk access

# Turning this on does lots of consistency checks to verify that the cache is accurate. This does slow down by about 35% though (w/245 components).
# With ~240 components, "noop" build took 2.8 sec without caching, and 0.8 sec with caching
CACHE_DEBUG = False

if CACHE_DEBUG:
  print "WARNING: filecache.CACHE_DEBUG is TRUE - this will slow down the build"


def Init():
  global _cache
  _cache = {}


class _Node(object):
  def __init__(self, exists, modify_time):
    self.exists = exists
    self.modify_time = modify_time


def _Fetch(path):
  if path not in _cache:
    metrics.fetch_file += 1
    metrics.exists_miss += 1
    if os.path.exists(path):
      metrics.get_modify_time_stat += 1
      _cache[path] = _Node(exists=True,
                           modify_time=os.stat(path)[stat.ST_MTIME])
    else:
      _cache[path] = _Node(exists=False,
                           modify_time=constants.MAX_AGE)

  return _cache[path]


def GetModifyTime(path):
  metrics.get_modify_time += 1
  node = _Fetch(path)
  if CACHE_DEBUG:
    if os.path.exists(path):
      assert node.modify_time == os.stat(path)[stat.ST_MTIME]
  return node.modify_time


def Delete(path):
  metrics.delete += 1
  node = _Fetch(path)
  if node.exists:
    metrics.delete_action += 1
    os.remove(path)
    node.exists = False
    node.modify_time = constants.MAX_AGE

  # DEBUG
  if CACHE_DEBUG:
    assert not os.path.exists(path)


def Exists(path):
  metrics.exists += 1
  node = _Fetch(path)
  if CACHE_DEBUG:
    assert node.exists == os.path.exists(path)
  return node.exists


def Purge(path):
  metrics.create += 1
  if path in _cache:
    del _cache[path]
