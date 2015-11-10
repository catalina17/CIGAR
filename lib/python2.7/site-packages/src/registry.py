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
# Stores components by name & module.
#
import os

import log
import util


def _StripModule(module):
  if module[:2] == '//':
    return module[2:]
  else:
    return module


class Registry(object):

  def __init__(self):
    self.modules = {}
    self.extensions = {}

  def Define(self, path, comp_type, **kwargs):
    """Create a new component, defined in a source file."""
    #print "Define %s (%s)" % (path, comp_type)
    dep = self.Get(path)
    if dep:
      reliants = ','.join([str(d) for d in dep.reliants])
      raise util.ConfigError('Component %s (%s) already defined by [%s]. Did you forget to include it as a component dependency?' % (path, comp_type, reliants))
    dep = comp_type(path, **kwargs)
    self.Register(dep)
    dep.SetDefined()  # TODO: Get rid of this?
    return dep

  def Reference(self, path, comp_type=None, define_missing=False, **kwargs):
    # TODO: Match also on kwargs, in case they are different.
    dep = self.Get(path)
    if not dep:
      if define_missing:
        if not comp_type:
          raise Exception('Cannot define missing reference, no type specified')
        #print "Creating missing ref: %s (%s)" % (path, comp_type)
        dep = self.Define(path, comp_type, **kwargs)

      else:
        raise util.ConfigError('Missing reference: %s' % path)

    assert dep
    if comp_type:
      if not isinstance(dep, comp_type):
        raise Exception('Expected %s to be of type %s, but is %s'
                        % (dep, comp_type, type(dep)))
    return dep

  def Register(self, comp):
    module = comp.GetDirectory()
    name = comp.GetShortName()
    #print "Module: %s, Name: %s, Comp: %s" % (module, name, comp)
    module = _StripModule(module)
    if name in self.modules[module]:
      raise util.ConfigError('Component %s already registered!"' % comp.DisplayName())
    #print "Register Component: %s:%s" % (module, name)
    self.modules[module][name] = comp

  def RegisterModule(self, module):
    module = _StripModule(module)
    if module in self.modules:
      raise Exception('Module %s already registered!' % module)
    #print "Register Module: %s" % module
    self.modules[module] = {}

  def Get(self, path):
    #print 'Get:', path
    if ':' in path:
      module, name = path.split(':')
    else:
      name = path
      module = ''
    if module not in self.modules:
      #TODO: This was commented out because PB buffers are in the /build/ path, which has
      # no registered module
      # raise util.ConfigError('Module %s not found!' % module)
      self.RegisterModule(module)
    if name in self.modules[module]:
      return self.modules[module][name]
    else:
      return None

  def GetTargets(self, target_name):
    # Possible trailing '/' in folder name (such as from command-line tab-completion)
    target_name = target_name.rstrip('/')
    if ':' in target_name:
      module_name, name = target_name.split(':')
      module = self.modules[module_name]
      if name:
        return [module[name]]
      else:
        return module.values()
    else:
      targets = []
      for module, comps in self.modules.items():
        if module.startswith(target_name):
          targets.extend(comps.values())
      return targets

  def GetAllTargets(self):
    comps = []
    for module in self.modules.values():
      comps.extend(module.values())
    return comps

  def RegisterExtension(self, name, comp_type, ext):
    if not comp_type in self.extensions:
      self.extensions[comp_type] = {}

    if name in self.extensions[comp_type]:
      raise Exception('Extension %s already defined for type %s!'
                      % (name, comp_type))

    #log.Info("Registered extension %s for type %s" % (name, comp_type))
    self.extensions[comp_type][name] = ext

  def GetExtension(self, name, comp_type):
    types = [comp_type]
    # breadth-first search up the class hierarchy
    while types:
      check_type = types[0]
      types = types[1:]
      if check_type in self.extensions:
        if name in self.extensions[check_type]:
          return self.extensions[check_type][name]
      types.extend(check_type.__bases__)
    return None

registry = Registry()  # Global instance
