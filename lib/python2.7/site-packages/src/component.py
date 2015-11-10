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
import os

import constants
import log
import metrics
from registry import registry
import util

import time   # temp debug


class Component(object):

  def __init__(self, name, **kwargs):
    self.name = name
    self.depends = set()
    self.reliants = set()
    self.kwargs = kwargs
    self.defined = False  # True when defined by a config file
    self.constructed = False  # True when 'Init' has been called.
    metrics.total_components += 1
    self.platform = None  # Suported platforms, None == all

  def __str__(self):
    return self.GetName()

  def Build(self):
    pass  # Default: no action

  def Run(self, args):
    pass  # Default: no action

  def Test(self):
    pass  # Default: no action

  def Clean(self):
    pass  # Default: no action

  def GetAllComponents(self, processed):
    self.OnInit()
    if self not in processed:
      processed.add(self)
      for dep in self.depends:
        dep.GetAllComponents(processed)

  def AllowCircularDepend(self, comp):
    # Default is to never allow circular deps. Derived components can override
    return False

  def AddDepend(self, comp):
    assert comp
    if self.HasReliant(comp) and not self.AllowCircularDepend(comp):
      raise util.ConfigError(
          "Circular dependency detected. %s is already reliant on %s"
          % (comp, self))

    self.depends.add(comp)
    comp.reliants.add(self)

  def HasReliant(self, comp, checked=None):
    if checked is None:
      checked = set()
    elif self in checked:
      return False
    checked.add(self)
    if self == comp:
      return True
    assert self != comp  # should be handled by set inclusuion check above

    # TODO: Cache all reliants for each component to avoid recursion
    for rel in self.reliants:
      if rel.HasReliant(comp, checked):
        return True
    return False

  def SetDefined(self):
    self.defined = True

  def IsDefined(self):
    return self.defined

  def GetPath(self):
    return util.NameToPath(self.name)

  def IsPlatformSupported(self):
    if self.platform:
      return util.GetPlatform() in self.platform
    else:
      return True

  def IsExclusiveTest(self):
    # Derived components can override to run test solo
    return False

  def GetName(self):
    return self.name

  def GetShortName(self):
    if ':' in self.name:
      pos = self.name.find(':') + 1
      return self.name[pos:]
    else:
      return self.name

  def GetDirectory(self):
    return os.path.dirname(self.GetPath())

  def GetBuildDirectory(self):
    return util.BuildPath(self.GetDirectory())

  def GetRecursiveModifyTime(self, checked=None):
    # TODO: This method goes away when i re-work how components/targets/outputs work ??
    if checked is None:
      checked = set()

    if self in checked:
      return constants.MAX_AGE
    else:
      checked.add(self)
      modify_time = self.GetModifyTime()
      if self.depends:
        youngest = max([d.GetRecursiveModifyTime(checked) for d in self.depends])
        modify_time = max(modify_time, youngest)
      return modify_time

  def Init(self, depend=None, platform=None, **kwargs):
    # TODO: Move this to a 'depend' extension?
    if depend:
      for d in self.ConvertPaths(depend):
        self.AddDepend(registry.Reference(d))
    self.platform = platform

  def OnInit(self):
    timer = metrics.Timer()
    timer.Start(resume=True)

    if not self.constructed:
      if not self.IsDefined():
        raise util.ConfigError('Component %s is not defined in an ATLAS file' % self)
      self.constructed = True
      #print "Init: %s" % self
      try:
        self.Init(**self.kwargs)
      except util.ConfigError, e:
        print "Defining %s" % self
        raise

      metrics.total_components_init += 1

    timer.Stop()
    metrics.analysis_timer.Add(timer)

  def OnBuild(self):
    assert self.constructed
    youngest = constants.MAX_AGE
    if self.depends:
      youngest = max([d.GetRecursiveModifyTime() for d in self.depends])
    if self.NeedsBuild(youngest):
      # 'Clean' so that old version doesn't exist on build failure
      self.OnClean()  # TODO: Check failure
      result = self.Build()
      return result
    else:
      return None

  def OnClean(self):
    assert self.constructed
    return self.Clean()

  def OnRun(self, args):
    assert self.constructed
    return self.Run(args)

  def OnTest(self):
    assert self.constructed
    if self.IsPlatformSupported():
      return self.Test()
    else:
      return None

  def ProcessExtensions(self, map):
    # Process Extensions First (may modify include_dirs)
    for key, values in map.items():
      ext = registry.GetExtension(key, type(self))
      if ext:
        if ext.convert_paths:
          values = self.ConvertPaths(values)
        for v in values:
          comp = registry.Reference(v, ext.comp_type)
          comp.OnInit()
          #log.Info('Processing extension %s for comp %s' % (key, comp))
          ext.Depend(comp, self)

  def ConvertPaths(self, paths):
    # Paths can be scalar string or a list of strings. We always return a list.
    results = util.ConvertPaths(paths, self.GetDirectory())
    if isinstance(results, list):
      return results
    else:
      return [results]
