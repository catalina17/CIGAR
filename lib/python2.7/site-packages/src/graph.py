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
# Atlas Dependency Graph
#
import re

import action
from component import Component
import constants
from flags import flags
import log
import metrics
from registry import registry
import util


class Graph(object):

  def __init__(self):
    pass

  def GetTargetComponents(self, targets):
    comps = []
    if targets:
      for target in targets:
        comps.extend(registry.GetTargets(target))
    else:
      comps.extend(registry.GetAllTargets())
    return comps

  def DependList(self, target_paths):
    all_comps = set()
    for comp in registry.GetAllTargets():
      comp.GetAllComponents(all_comps)

    for path in target_paths:
      target_comp = registry.Get(path)
      if target_comp:
        matches = []
        for comp in all_comps:
          if comp != target_comp and comp.HasReliant(target_comp):
            matches.append(comp)
        print '%s has %d dependencies: ' % (target_comp, len(matches))
        for m in sorted(matches, key=Component.__str__):
          print "\t%s" % m
      else:
        print "Bad target: ", path
        return False
    return True

  def ReliantList(self, targets):
    print 'Reliants...'
	# TODO(mparent)
    pass

  def Build(self, targets):
    print 'Building...'
    failed = []
    def OnComplete(comp, result):
      if result.success:
        if flags.verbose and result.command:
          print '\t', result.command
        else:
          print '\t', result.desc
        if result.output:
          for line in result.output: print line
          print  # empty line
      else:
        failed.append(result)
        if result.timeout:
          util.ColorPrint('\n<RED>Timeout: </RED>%s' % result.desc)
        else:
          util.ColorPrint('\n<RED>Error: </RED>%s' % result.desc)
        if flags.verbose:
          print 'Command:', result.command
        for line in result.output: print line
        return False

    action.Work(self.GetTargetComponents(targets),
                Component.OnBuild,
                OnComplete)

    if failed:
      util.ColorPrint('<RED>BUILD FAILED</RED>')
      return False
    else:
      if flags.verbose: util.ColorPrint('BUILD SUCCEEDED')
      return True

  def Rebuild(self, targets):
    # Convenience method
    return self.Clean(targets) and self.Build(targets)

  def Clean(self, targets):
    print 'Cleaning...'
    def OnComplete(comp, result):
      if flags.verbose:
        print '\t', result.desc

    action.Work(self.GetTargetComponents(targets),
                Component.OnClean,
                OnComplete)
    # TODO: It is possible that clean could fail!
    return True #result.Success()

  def Run(self, targets):
    run_target = targets[:1]
    run_args = targets[1:]
    if not self.Build(run_target):
      return False
    print "Running..."

    success = True
    def OnComplete(comp, result):
      if not result.success:
        success = False
      for line in result.output:
        print line

    def OnRun(comp):
      comp.OnRun(run_args)

    action.Work(self.GetTargetComponents(run_target),
                OnRun,
                OnComplete)

    return success

  def Test(self, targets):
    if not self.Build(targets):
      return False
    print "Testing..."
    passed = []
    failed = []
    def OnComplete(comp, result):
      if result.success:
        passed.append(result)
        util.ColorPrint('\t<GREEN>PASS</GREEN>\t%s' % result.desc)
#for line in result.output: print line
      else:
        failed.append(result)
        if result.timeout:
          util.ColorPrint('\t<RED>TIMEOUT</RED>\t%s' % result.desc)
        else:
          util.ColorPrint('\t<RED>FAIL</RED>\t%s' % result.desc)
        print '-' * 50
        for line in result.output[-constants.MAX_TEST_OUTPUT_LINES:]:
          print line
        print '-' * 50

    action.Work(self.GetTargetComponents(targets),
                Component.OnTest,
                OnComplete)

    if failed:
      count = len(failed)
      util.ColorPrint('\n\t<RED>%d TEST%s FAILED</RED>'
                      % (count, util.Plural(count, 'S')))
      for test in failed:
        print "\t\t%s" % test.desc
      return False
    else:
      count = len(passed)
      util.ColorPrint('\n\t<GREEN>%d TEST%s PASSED</GREEN>'
                      % (count, util.Plural(count, 'S')))
      return True
