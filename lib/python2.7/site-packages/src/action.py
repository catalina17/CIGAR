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
# action.py - Queue up actions and run in parallel (TODO)
#
import os
import signal
import subprocess
import tempfile
import threading
import threadpool
import time
import traceback

import constants
from flags import flags
import log
import util


TIMEOUT = 0
SUCCESS = 1
FAILURE = 2


class CommandError(Exception):
  pass


class Result(object):
  def __init__(self, result, output, desc, command=None):
    if result == SUCCESS:
      self.success = True
      self.timeout = False
    else:
      self.success = False
      self.timeout = (result == TIMEOUT)
    self.output = output
    self.desc = desc
    self.command = command


class Queue(object):
  def __init__(self):
    self.checked = set()  # Processed by 'populate'
    self.queue = []
    self.scheduled = set()  # Added to self.queue
    self.pending = set()
    self.completed = set()
    self.map = {}           # Maps dependent to direct parents
    self.stop = False

  def Stop(self):
    self.stop = True

  def Finished(self):
    return not self.pending or self.stop

  def Next(self, exclusive_callback):
    if self.stop:
      return []
    else:
      next = self.queue
      self.queue = []
      return next

  def Schedule(self, comp):
    assert not comp in self.scheduled
    self.scheduled.add(comp)
    self.queue.append(comp)

  def Map(self, depend, parent):
    #print "Map %s --> %s" % (depend, parent)
    if not depend in self.map:
      self.map[depend] = set()
    self.map[depend].add(parent)

  def Populate(self, comp):
    if not comp in self.checked:
      self.checked.add(comp)
      self.pending.add(comp)
      comp.OnInit()

      # If we find a circular dependency, # then don't treat as a
      # build dependency.  We trust the graph builder to only allow
      # valid circular dependencies.
      has_build_depends = False
      if comp.depends:
        for dep in comp.depends:
          if not comp.HasReliant(dep):
            has_build_depends = True
            self.Map(dep, comp)

          self.Populate(dep)

      if not has_build_depends and comp not in self.scheduled:
        #print "Queueing (leaf) %s" % comp
        self.Schedule(comp)

  def Complete(self, comp):
    assert comp not in self.completed
    #print "Completed: %s" % comp
    self.completed.add(comp)
    self.pending.remove(comp)
    if comp in self.map:
      for parent in self.map[comp]:
        if not parent in self.scheduled:
          # Make sure all dependencies are built
          ready = True
          for dep in parent.depends:
            if dep not in self.completed and not parent.HasReliant(dep):
              ready = False
          if ready:
            #print 'Queuing (has all depends): ', parent
            self.Schedule(parent)


def Work(targets, action, callback=None, exclusive_callback=None):

  def Done(request, result):
    comp = request.args[0]
    queue.Complete(comp)
    if callback:
      if not isinstance(result, list):
        result = [result]
      for r in result:
        if r and callback(comp, r) == False:
          queue.Stop()
    Push()

  def Except(request, except_info):
    traceback.print_exception(*except_info)
    raise CommandError('Error running worker task')

  if flags.parallel:
    # TODO: Might want even more processors if we are using distcc...
    processor_count = util.GetProcessorCount()
    log.Info("Found %d processors" % processor_count)
    workers = processor_count * constants.WORKERS_PER_PROCESSOR
    log.Info('[Parallel Mode - %d workers]' % workers)
  else:
    workers = 1
    print '[Serial Mode]'
  pool = threadpool.ThreadPool(workers)

  def Push():
    next = queue.Next(exclusive_callback)
    for req in threadpool.makeRequests(action, next, Done, Except):
      pool.putRequest(req)

  queue = Queue()
  for comp in targets:
    queue.Populate(comp)
  Push()
  # Need 'while' loop in case 'wait()' exits due to empty queue, but we are in the process of adding more requests.
  while not queue.Finished():
    pool.wait()
    time.sleep(0.1)


class Command(object):
  def __init__(self):
    pass


class _ReadPipeThread (threading.Thread):
  """Allows non-blocking read() calls to sub-process's 'stdout' pipes"""
  def __init__(self, source, buffered=True):
    threading.Thread.__init__(self)
    self.source = source
    self.buffered = buffered

  def run (self):
    if self.buffered:
      # This will block until we hit EOF (ie the sub-process returns)
      # This way is faster for build/test, which only prints output when
      # command completes.
      self.output = self.source.readlines()
    else:
      # Slower, but prints output immediately (for 'run' command).
      self.output = []
      while True:
        line = self.source.readline()
        if not line:
          return
        self.output.append(line)
        print line,


command_lock = threading.Lock()
class LockedProcess(object):
  def __init__(self, buffered, *args, **kwargs):
    #command_lock.acquire()
    self.process = subprocess.Popen(*args, **kwargs)
    self.reader = _ReadPipeThread(self.process.stdout, buffered)
    self.reader.start()
    #command_lock.release()

  def Poll(self):
    #command_lock.acquire()
    result = self.process.poll()
    #command_lock.release()
    return result

  def Kill(self):
    os.kill(self.process.pid, signal.SIGKILL)
     #TODO: Kill all child procs, what about WNOHANG????
    os.waitpid(self.process.pid, 0) #os.WNOHANG)

  def ReturnCode(self):
    return self.process.returncode

  def Output(self):
    self.reader.join()
    output = [line.rstrip() for line in self.reader.output]
#TODO(mparent): Temporily disabled to track down a crash.
#assert self.process.returncode is not None
    return_code = self.process.returncode
    if return_code is not None and return_code < 0:
      signal_num = -(self.process.returncode)
      name, desc = util.DescribeSystemSignal(signal_num)
      output.append('Terminated: %s - %s' % (name, desc))
    return output

class ShellCommand(Command):
  def __init__(self, action, desc, work_dir, timeout=None, args=None, buffered=True):
    Command.__init__(self)
    self.desc = desc
    path = action
    # Handle targets in CWD - need './action'
    if not os.path.sep in path and util.Exists(path):
      path = os.path.join('.', path)
    self.target = path
    self.args = [self.target]
    if args:
      self.args.extend(args)
    self.timeout = timeout
    self.work_dir = util.AbsPath(work_dir)
    self.buffered = buffered

  def Add(self, raw):
    self.args.append(raw)

  def Extend(self, args):
    self.args.extend(args)

  def AddFlags(self, flags):
    if flags:
      self.args.extend(flags)

  def AddPath(self, path):
    self.args.append(util.RelPath(path))

  def AddParamPath(self, param, path):
    self.args.append(param)
    self.args.append(util.RelPath(path))

  def _RawCommand(self):
    return ' '.join(self.args)

  def _Shell(self):
    start = time.time()

    # Provide programs with some common root paths.
    # TODO: Should I just have a master set of vars?
    # TODO: If this is only used by 'Tests', should I move this?
    env = os.environ.copy()
    env['ATLAS_ROOT_DIR'] = util.GetRootDirectory()
    env['ATLAS_BUILD_DIR'] = os.path.join(util.GetRootDirectory(), util.GetBuildDirectory())

    # The 'exec' command prevents shell from forking, so that killing process on timeout kills off the shell AND the actual command child process
    # TODO: If sub-command spawns off child procs, are we kiling those too?
    command = 'exec ' + self._RawCommand()
    process = LockedProcess(args=command,
                            cwd=self.work_dir,
                            buffered=self.buffered,
                            env=env,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,  # Pipe everyting through stdout
                            shell=True)  # Need shell to run 'make', 'gcc' etc.
    while process.Poll() is None:
      now = time.time()
      if self.timeout is not None and (now - start) > self.timeout:
        log.Info("Timeout Triggered! (%s sec)" % self.timeout)
        process.Kill()
        time.sleep(1)
        return TIMEOUT, process.Output()
      time.sleep(1)
    assert process.ReturnCode() is not None

    if process.ReturnCode():
      return FAILURE, process.Output()
    else:
      return SUCCESS, process.Output()


  def Run(self):
    result, output = self._Shell()
    return Result(result, output, self.desc, self._RawCommand())


class BuildCommand(ShellCommand):
  def __init__(self, action, desc=None, timeout=None, work_dir=None):
    if timeout is None:
      timeout = constants.BUILD_TIMEOUT
    if work_dir is None:
      work_dir = os.getcwd()
    ShellCommand.__init__(self, action, desc=desc, timeout=timeout,
                          work_dir=work_dir)


class TestCommand(ShellCommand):
  def __init__(self, action, desc=None, timeout=None):
    if timeout is None:
      timeout = constants.TEST_TIMEOUT
    if desc is None:
      desc = util.StripBuildDirectory(util.RelPath(action))
    ShellCommand.__init__(self, action, desc=desc, timeout=timeout,
                          work_dir=tempfile.mkdtemp(prefix='atlas_test'))


class RunCommand(ShellCommand):
  def __init__(self, target, desc=None):
    if desc is None:
      desc = util.StripBuildDirectory(util.RelPath(target))
    util.SetExecutePermission(target)
    ShellCommand.__init__(self, target, desc=desc, work_dir=os.getcwd(),
                          buffered=False)


class DeleteCommand(Command):
  def __init__(self, target):
    Command.__init__(self)
    self.target = target

  def Run(self):
    if util.Exists(self.target):
      success = util.Delete(self.target)
      if success:
        # Now delete any empty parent directories up to (& including) 'build/' root.
        dir = os.path.dirname(self.target)
        while dir:
          assert dir.startswith(constants.BUILD_ROOT)  # sanity check
          try:
            os.rmdir(dir)
          except OSError:
            break  # Directory is not empty, stop.
          dir = os.path.dirname(dir)
        result = SUCCESS
      else:
        result = FAILURE
      return Result(result, [], 'deleted %s' % self.target)
