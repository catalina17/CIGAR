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
import re
import os

from src import log
from src.registry import registry
from src import util

from src.mod.file import FileComponent


HEADER_INCLUDE_RE = re.compile('^\s*#include\s+"([^"]+)"', re.M)

class IncludeError(util.ConfigError):
  def __init__(self, msg, include_comp):
    msg += '\n\tDependent File: ' + include_comp.GetPath()
    msg += '\n\tInclude Directories:'
    for dir in sorted(include_comp.include_dirs):
      msg += '\n\t\t' + util.RelPath(dir)
    util.ConfigError.__init__(self, msg)


class CppIncludeComponent(FileComponent):
  """Represents an include file (likely *.h) dependency."""

  def Init(self, include_dirs=None, make_depends=True):
    FileComponent.Init(self)
    self.include_dirs = set(include_dirs or [])

    if make_depends:
      self.MakeDepends()

  def AddIncludeDirectory(self, dir):
    dir = util.RelPath(dir)
    self.include_dirs.add(dir)

  def AllowCircularDepend(self, comp):
    # Allow circular headers. Compiler (gcc) will detect errors.
    return isinstance(comp, CppIncludeComponent)

  def MakeDepends(self):
    for include in util.GetFileMatches(self.source, HEADER_INCLUDE_RE, 'CPP include'):
      count = 0
      for d in self.include_dirs:
        full_include = os.path.join(d, include)
        comp = registry.Get(util.PathToName(full_include))
        exists = util.Exists(full_include)
        if comp or exists:
          count += 1
          comp = registry.Reference(util.PathToName(full_include),
                                    CppIncludeComponent,
                                    define_missing=True,
                                    include_dirs=self.include_dirs)
          if self.HasReliant(comp):
            log.Warning('Header %s included by %s: Circular header dependency detected.' % (full_include, self.GetPath()))
          #else:
          self.AddDepend(comp)
      if not count:
        raise IncludeError('Did not find any matches for header file: %s' % include, self)
      elif count > 1:
        for d in self.include_dirs:
          full_include = os.path.join(d, include)
          if util.Exists(full_include):
            print "POSSIBLE HEADER: %s" % full_include
        raise IncludeError('Found multiple possible files for header file "%s"' % include, self)
