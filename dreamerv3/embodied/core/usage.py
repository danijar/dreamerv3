import gc
import inspect
import os
import re
import threading
import time
import tracemalloc
from collections import defaultdict

from . import agg
from . import timer


class Usage:

  def __init__(self, **kwargs):
    available = {
        'psutil': PsutilStats,  # per process and global
        'nvsmi': NvsmiStats,    # gloal
        'gputil': GputilStats,  # per process
        'malloc': MallocStats,  # per process
        'gc': GcStats,          # per process
    }
    self.tools = {}
    for name, enabled in kwargs.items():
      assert isinstance(enabled, bool), (name, type(enabled))
      if enabled:
        self.tools[name] = available[name]()

  def stats(self):
    stats = {}
    for name, tool in self.tools.items():
      stats.update({f'{name}/{k}': v for k, v in tool().items()})
    return stats


class NvsmiStats:

  PATTERNS = {
      'compute_min': (r'GPU Utilization Samples(.|\n)+?Min.*?: (\d+) %', 1),
      'compute_avg': (r'GPU Utilization Samples(.|\n)+?Avg.*?: (\d+) %', 1),
      'compute_max': (r'GPU Utilization Samples(.|\n)+?Max.*?: (\d+) %', 1),
      'memory_min': (r'Memory Utilization Samples(.|\n)+?Min.*?: (\d+) %', 1),
      'memory_avg': (r'Memory Utilization Samples(.|\n)+?Avg.*?: (\d+) %', 1),
      'memory_max': (r'Memory Utilization Samples(.|\n)+?Max.*?: (\d+) %', 1),
  }

  def __init__(self):
    pass

  @timer.section('nvsmi_stats')
  def __call__(self):
    output = os.popen('nvidia-smi --query -d UTILIZATION 2>&1').read()
    if not output:
      print('To log GPU stats, make sure nvidia-smi is working.')
      return {}
    metrics = {'output': output}
    for name, (pattern, group) in self.PATTERNS.items():
      numbers = [x[group] for x in re.findall(pattern, output)]
      for i, number in enumerate(numbers):
        metrics[f'{name}/gpu{i}'] = float(numbers[i]) / 100
    return metrics


class PsutilStats:

  def __init__(self):
    import psutil
    self.proc = psutil.Process()

  @timer.section('psutil_stats')
  def __call__(self):
    import psutil
    gb = 1024 ** 3
    cpus = psutil.cpu_count()
    memory = psutil.virtual_memory()
    stats = {
        'proc_cpu_usage': self.proc.cpu_percent() / 100,
        'proc_ram_frac': self.proc.memory_info().rss / memory.total,
        'proc_ram_gb': self.proc.memory_info().rss / gb,
        'total_cpu_count': cpus,
        'total_cpu_frac': psutil.cpu_percent() / 100,
        'total_ram_frac': memory.percent / 100,
        'total_ram_total_gb': memory.total / gb,
        'total_ram_used_gb': memory.used / gb,
        'total_ram_avail_gb': memory.available / gb,
    }
    return stats


class GputilStats:

  def __init__(self):
    import GPUtil
    self.gpus = GPUtil.getGPUs()
    print(f'GPUtil found {len(self.gpus)} GPUs')
    self.error = None
    self.aggs = defaultdict(agg.Agg)
    self.once = True
    self.worker = threading.Thread(target=self._worker, daemon=True)
    self.worker.start()

  @timer.section('gputil_stats')
  def __call__(self):
    if self.error:
      raise self.error
    stats = {}
    for i, agg_ in self.aggs.items():
      stats.update(agg_.result(prefix=f'gpu{i}'))
    if self.once:
      self.once = False
      lines = [f'GPU {i}: {gpu.name}' for i, gpu in enumerate(self.gpus)]
      stats['summary'] = '\n'.join(lines)
    return stats

  def _worker(self):
    try:
      while True:
        for i, gpu in enumerate(self.gpus):
          agg = self.aggs[i]
          agg.add('load', gpu.load, 'avg')
          agg.add('mem_free_gb', gpu.memoryFree / 1024, 'min')
          agg.add('mem_used_gb', gpu.memoryFree / 1024, 'max')
          agg.add('mem_total_gb', gpu.memoryTotal / 1024)
          agg.add('memory_util', gpu.memoryUtil, ('min', 'avg', 'max'))
          agg.add('temperature', gpu.temperature, 'max')
        time.sleep(0.5)
    except Exception as e:
      print(f'Exception in Gputil worker: {e}')
      self.error = e


class GcStats:

  def __init__(self):
    gc.callbacks.append(self._callback)
    self.stats = agg.Agg()
    self.keys = set()
    self.counts = [{}, {}, {}]
    self.start = None

  @timer.section('gc_stats')
  # def __call__(self, log=False):
  def __call__(self, log=True):
    stats = {k: 0 for k in self.keys}
    stats.update(self.stats.result())
    stats['objcounts'] = self._summary()
    log and print(stats['objcounts'])
    self.keys |= set(stats.keys())
    return stats

  def _summary(self):
    lines = ['GC Most Common Types']
    for gen in range(3):

      objs = {
          id(obj): obj for obj in gc.get_objects(gen)
          if not inspect.isframe(obj)}
      for obj in list(objs.values()):
        for obj in gc.get_referents(obj):
          if not gc.is_tracked(obj):
            objs[id(obj)] = obj

      counts = defaultdict(int)
      for obj in objs.values():
        counts[type(obj).__name__] += 1

      deltas = {k: v - self.counts[gen].get(k, 0) for k, v in counts.items()}
      self.counts[gen] = counts

      deltas = dict(sorted(deltas.items(), key=lambda x: -abs(x[1]))[:10])
      lines.append(f'\nGeneration {gen}\n')
      for name, delta in deltas.items():
        lines.append(f'- {name}: {delta:+d} ({counts[name]})')

    return '\n'.join(lines)

  def _callback(self, phase, info):
    # We cannot wrap this function into a timer section, because it would get
    # nested into an arbitrary scope that was active before the garbage
    # collector got triggered.
    now = time.perf_counter_ns()
    if phase == 'start':
      self.start = now
    if phase == 'stop' and self.start:
      gen = info['generation']
      agg = ('avg', 'max', 'sum')
      self.stats.add(f'gen{gen}/calls', 1, agg='sum')
      self.stats.add(f'gen{gen}/collected', info['collected'], agg)
      self.stats.add(f'gen{gen}/uncollectable', info['collected'], agg)
      self.stats.add(f'gen{gen}/duration', (now - self.start) / 1e9, agg)


class MallocStats:

  def __init__(self):
    tracemalloc.start()
    self.previous = None

  @timer.section('malloc_stats')
  def __call__(self, log=True):
    stats = {}
    snapshot = tracemalloc.take_snapshot()
    stats['full'] = self._summary(snapshot)
    stats['diff'] = self._summary(snapshot, self.previous)
    self.previous = snapshot
    log and print(stats['full'])
    return stats

  def _summary(self, snapshot, relative=None, top=50, root='embodied'):
    if relative:
      statistics = snapshot.compare_to(relative, 'traceback')
    else:
      statistics = snapshot.statistics('traceback')
    agg = defaultdict(lambda: [0, 0])
    for stat in statistics:
      filename = stat.traceback[-1].filename
      lineno = stat.traceback[-1].lineno
      for frame in reversed(stat.traceback):
        if f'/{root}/' in frame.filename:
          filename = f'{root}/' + frame.filename.split(f'/{root}/')[-1]
          lineno = frame.lineno
          break
      agg[(filename, lineno)][0] += stat.size_diff if relative else stat.size
      agg[(filename, lineno)][1] += stat.count_diff if relative else stat.count
    lines = []
    lines.append('\nMemory Allocation' + (' Changes' if relative else ''))
    lines.append(f'\nTop {top} by size:\n')
    entries = sorted(agg.items(), key=lambda x: -abs(x[1][0]))
    for (filename, lineno), (size, count) in entries[:top]:
      size = size / (1024 ** 2)
      lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')
    lines.append(f'\nTop {top} by count:\n')
    entries = sorted(agg.items(), key=lambda x: -abs(x[1][1]))
    for (filename, lineno), (size, count) in entries[:top]:
      size = size / (1024 ** 2)
      lines.append(f'- {size:.2f}Mb ({count}) {filename}:{lineno}')
    return '\n'.join(lines)
