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
# Performance Metrics
#
import time

import constants
from flags import flags


class Timer(object):
  def __init__(self):
    self.elapsed = 0
    self.running = False

  def Start(self, resume=False):
    if resume:
      assert not self.running
    else:
      self.elapsed = 0
    self.start = time.time()
    self.running = True

  def Stop(self):
    assert self.running
    self.elapsed += (time.time() - self.start)
    self.running = False

  def Elapsed(self):
    assert not self.running
    return self.elapsed

  def Add(self, timer):
    self.elapsed += timer.elapsed

# Timers
action_timer = Timer()
analysis_timer = Timer()
load_timer = Timer()
total_timer = Timer()

# Method Counts
get_modify_time = 0
get_modify_time_stat = 0
delete = 0
delete_action = 0
create = 0
exists = 0
exists_miss = 0
fetch_file = 0
relpath = 0
build_path = 0

# Project Stats
config_files = 0
total_modules = 0
total_components = 0
total_components_init = 0

def Output():
  s = ''

  s += '\nElapsed Time:'
  s += '\n\tLoad       %6.1fs' % load_timer.Elapsed()
  s += '\n\tAnalysis   %6.1fs' % analysis_timer.Elapsed()
  s += '\n\tAction     %6.1fs' % action_timer.Elapsed()
  s += '\n\tTotal      %6.1fs' % total_timer.Elapsed()

  s += '\n\nProject Stats:'
  s += '\n\tconfig files             %d' % config_files
  s += '\n\ttotal modules            %d' % total_modules
  s += '\n\ttotal components         %d' % total_components
  s += '\n\ttotal components init    %d' % total_components_init

  s += '\n\nMethod Calls:'
  s += '\n\tDelete                     %d' % delete
  s += '\n\tDelete Action              %d' % delete_action
  s += '\n\tCreate                     %d' % create
  s += '\n\tFetch                      %d' % fetch_file
  s += '\n\tExists                     %d' % exists
  s += '\n\tExists Miss                %d' % exists_miss
  s += '\n\tutil.GetModifyTime         %d' % get_modify_time
  s += '\n\tutil.GetModifyTime (stat)  %d' % get_modify_time_stat
  s += '\n\tutil.RelPath               %d' % relpath
  s += '\n\tutil.BuildPath             %d' % build_path

  return s

def Report(command, targets):
  if flags.metrics:
    print Output()

  out = ''
  out += '\n' + ('-' * 40)
  out += '\n' + time.asctime(time.localtime())
  out += '\nATLAS v' + constants.VERSION
  out += '\nCommand: ' +  command
  out += '\nTargets: %s' %  targets
  out += Output() 

  file = open('.atlas_metrics.log', 'a')
  file.write(out)
  
