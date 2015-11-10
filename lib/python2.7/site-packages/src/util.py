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
import commands
import glob
import platform
import os
import re
import stat
import threading

import constants
import filecache
from flags import flags
import log
import metrics

COLOR_CODES = {'red': '01;31',
               'green': '0;32',
               'blue': '01;34'
              }

PLATFORM_OSX = 'osx'
PLATFORM_LINUX = 'linux'
PLATFORM_DEFAULT = PLATFORM_LINUX


def ColorPrint(s, newline=True):
  #TODO: Support all platforms.
  #TODO: Support more colors, use some sort of hash/lookup table
  if flags.color:
    s = s.replace('<RED>', '\033[01;31m')
    s = s.replace('<GREEN>', '\033[0;32m')
    s = s.replace('<BLUE>', '\033[01;34m')
    s = re.sub('</[a-zA-Z]+>', '\033[m', s)
  else:
    s = re.sub('</?[a-zA-Z]+>', '', s)

  if newline:
    print s
  else:
    print s,


def ResetPrintLine():
  print '\033[1A',  # Move cursor up 1 line
  print '\033[K',   # Erase line contents


def Plural(count, ending='s'):
  if count == 1:
    return ''
  else:
    return ending


def CompareName(a, b):
  return cmp(a.GetName(), b.GetName())


def MakeParentDir(path):
  MakeDir(os.path.dirname(path))


# TODO: This lock is not ideal...
makedir_lock = threading.Lock()
def MakeDir(path):
  makedir_lock.acquire()
  _MakeDir(path)
  makedir_lock.release()


def _MakeDir(path):
  if os.path.isdir(path):
    pass
  elif os.path.isfile(path):
    raise Exception('File with the same name as target dir already exists: %s' % path)
  else:
    head, tail = os.path.split(path)
    if head and not os.path.isdir(head):
      _MakeDir(head)
    if tail:
      os.mkdir(path)


class Cache(object):
  """Basic cache class. The problem is that I need to clear the cache after cleaning. But this could definitely speed things up."""
  def __init__(self):
    self.metric_hits = 0
    self.metric_misses = 0
    self.store = {}

  def Get(self, key, op):
    if key in self.store:
      self.metric_hits += 1
    else:
      self.metric_misses += 1
      self.store[key] = op(key)
    return self.store[key]


class ConfigError(Exception):
  pass


def GetModifyTime(path):
  return filecache.GetModifyTime(path)


def Delete(path):
  filecache.Delete(path)
  return True


def Exists(path):
  return filecache.Exists(path)


def GetModule(path):
  if path == '.' or not path:
    return '//'
  else:
    return '//' + path


def AbsPath(path):
  return os.path.abspath(path)


def RelPath(path):
  metrics.relpath += 1
  # TODO: make me faster, perhaps use system call?
  if path == os.getcwd():
    return '.'
  elif path.startswith('./'):
    return path[2:]
  else:
    # TODO: Make sure cwd part is at beginning of string.
    return path.replace(os.getcwd() + '/', '')


def GetRootDirectory():
  return os.getcwd()


def GetBuildDirectory():
  if flags.debug:
    return constants.DEBUG_BUILD_DIR
  else:
    return constants.OPT_BUILD_DIR


def BuildPath(path=None):
  metrics.build_path += 1
  if not path or path == '.':
    return GetBuildDirectory()
  elif (path.startswith(GetBuildDirectory()) or
        path.startswith(os.path.abspath(GetBuildDirectory()))):
    return path  # Already build-relative
  elif path[0] == '/':
    return os.path.abspath(BuildPath(RelPath(path)))
  else:
    return os.path.join(GetBuildDirectory(), path)


def StripBuildDirectory(path):
  return path.replace(GetBuildDirectory() + '/', '')


def IsPlatform(type):
  #print "Check platform: %s" % type
  type = type.lower()
  if type == 'all':
    return True
  else:
    return type == GetPlatform()


def GetPlatform():
  """Return string describing current OS platform."""
  if bool(platform.mac_ver()[0]):
    return PLATFORM_OSX
  elif bool(platform.dist()[0]):
    return PLATFORM_LINUX
  log.Error('Error: Unknown platform type: %s' % type)
  return PLATFORM_DEFAULT


# Helper used to force ATLAS arguments into list form.
def MakeArgumentList(args):
  if args is None:
    arg_list = []
  elif isinstance(args, str):
    arg_list = [args]
  else:
    assert isinstance(args, list)
    arg_list = args

  filtered = []
  for idx in range(len(arg_list)):
    if isinstance(arg_list[idx], str):
      filtered.append(arg_list[idx])
    elif isinstance(arg_list[idx], tuple):
      value, allowed_platform = arg_list[idx]
      if IsPlatform(allowed_platform):
        filtered.append(value)
    else:
      log.Error('Unknown argument type')
  return filtered


def GlobPath(path, base_dir):
  path = os.path.join(base_dir, path)
  paths = glob.glob(path)
  offset = len(base_dir)
  return [p[offset:] for p in paths]


def ConvertPaths(paths, base_dir):
  if isinstance(paths, str):
    return _ConvertPath(paths, base_dir)
  else:
    return [_ConvertPath(p, base_dir) for p in (paths or [])]


def _ConvertPath(path, base_dir):
  if path.startswith('//'):
    assert ':' in path
    return path[2:]
  else:
    if base_dir:
      if ':' in path:
        return '%s/%s' % (base_dir, path)
      elif '/' in path:
        path = '%s/%s' % (base_dir, path)
        return PathToName(path)
      else:
        return '%s:%s' % (base_dir, path)
    else:
      return path


def NameToPath(name):
  return name.replace(':', '/')


def PathToName(path):
  if path.startswith('./'):
    path = path[2:]
  pos = path.rfind('/')
  if pos is not None:
    return path[:pos] + ':' + path[pos+1:]  # TODO: UGLY!
  else:
    return path


def GetFileMatches(filename, re_pattern, desc):
  """Get all regex matches contained in a file.

     Used to find CPP include files, etc.
  """
  try:
    content = open(filename).read()
  except IOError, e:
    raise ConfigError('Error opening %s file %s: %s' % (desc, filename, e))

  return re_pattern.findall(content)


def SetExecutePermission(path):
  """Set execute permission on a file, if necessary."""
  mode = os.stat(path).st_mode
  if not (mode & stat.S_IEXEC):
    os.chmod(path, (mode | stat.S_IEXEC))


def GetProcessorCount():
   """
   Detects the number of CPUs on a system. Cribbed from pp.
   From http://codeliberates.blogspot.com/2008/05/detecting-cpuscores-in-python.html
   """
   # Linux, Unix and MacOS:
   if hasattr(os, 'sysconf'):
       if os.sysconf_names.has_key('SC_NPROCESSORS_ONLN'):
           # Linux & Unix:
           ncpus = os.sysconf('SC_NPROCESSORS_ONLN')
           if isinstance(ncpus, int) and ncpus > 0:
               return ncpus
       else: # OSX:
           return int(os.popen2('sysctl -n hw.ncpu')[1].read())
   # Windows:
   if os.environ.has_key('NUMBER_OF_PROCESSORS'):
           ncpus = int(os.environ['NUMBER_OF_PROCESSORS']);
           if ncpus > 0:
               return ncpus
   return 1  # Default


def DescribeSystemSignal(signal_num, platform=None):

  if platform is None:
    platform = GetPlatform()

  if platform == PLATFORM_OSX:
    signal_map = {
        # TODO: Add more signals
        1:  ('SIGHUP', 'Hangup detected on controlling terminal or death of controlling process'),
        2:  ('SIGINT', 'Interrupt from keyboard'),
        4:  ('SIGILL', 'Illegal Instruction'),
        6:  ('SIGABRT', 'Abort signal from abort()'),
        8:  ('SIGFPE', 'Floating point exception'),
        10: ('SIGBUS', 'Bus Error (bad memory access)'),
        11: ('SIGSEGV', 'Invalid memory reference'),
        13: ('SIGPIPE', 'Broken pipe: write to pipe with no readers'),
        15: ('SIGTERM', 'Termination signal'),
    }
  elif platform == PLATFORM_LINUX:
    signal_map = {
        7: ('SIGBUS', 'Bus Error (bad memory access)'),
        9:  ('SIGKILL', 'Illegal Instruction'),
       11:  ('SIGSEGV', 'Segmentation Violation'),
    }

  else:
    raise 'Unhandled platform!'

  if signal_num in signal_map:
    return signal_map[signal_num]
  else:
    return ('SIGNAL %d' % signal_num, 'Unknown signal!')
