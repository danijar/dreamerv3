import argparse
import collections
import functools
import gzip
import json
import multiprocessing as mp
import pathlib
import re
import subprocess
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import rich.console
import tqdm

TITLES = {
    'dmlab_explore_goal_locations_small': 'DMLab Goals Small',
    'crafter_reward': 'Crafter',
    'pinpad2_three': 'Pin Pad Three',
    'pinpad2_four': 'Pin Pad Four',
    'pinpad2_five': 'Pin Pad Five',
    'pinpad2_six': 'Pin Pad Six',
    'pinpad2_eight': 'Pin Pad Eight',
    'loconav_ant_maze_s_50hz': 'Ant Maze S',
    'loconav_ant_maze_m_50hz': 'Ant Maze M',
    'loconav_ant_maze_l_50hz': 'Ant Maze L',
    'loconav_ant_maze_xl_50hz': 'Ant Maze XL',
}

COLORS = {
    'contrast': (
        '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
        '#001177', '#117700', '#990022', '#885500', '#553366', '#006666'),
    'gradient': (
        '#a0da39', '#4ac16d', '#277f8e', '#365c8d', '#46327e', '#440154'),
    'gradient_more': (
        '#fde725', '#a0da39', '#4ac16d', '#1fa187', '#277f8e', '#365c8d',
        '#46327e', '#440154'),
}


def main():
  console = rich.console.Console()
  args = parse_args()
  runs = []
  for directory in args.indirs:
    seed_prefix = len(args.indirs) > 1 and directory.name
    method_prefix = args.prefix and directory.name
    runs += load_metrics(
        directory, args.pattern, args.xaxis, args.yaxis, args.yaxis2,
        seed_prefix, method_prefix, args.tasks, args.methods, args.workers)
  tasks = []
  for regex in args.tasks:
    found = [x['task'] for x in runs if re.search(regex, x['task'])]
    [tasks.append(x) for x in natsort(found) if x not in tasks]
  methods = []
  for regex in args.methods:
    found = [x['method'] for x in runs if re.search(regex, x['method'])]
    [methods.append(x) for x in natsort(found) if x not in methods]
  seeds = natsort(set(run['seed'] for run in runs))
  console.print(f'Tasks ({len(tasks)}): [cyan]{", ".join(tasks)}[/cyan]')
  console.print(f'Methods ({len(methods)}): [cyan]{", ".join(methods)}[/cyan]')
  console.print(f'Seed ({len(seeds)}): [cyan]{", ".join(seeds)}[/cyan]')
  if not runs:
    console.print('Nothing to plot!', style='red')
    return
  args.outdir.mkdir(parents=True, exist_ok=True)

  if args.stats:
    print('Computing stats...', flush=True)
    len(tasks) == 1 and 'mean' in args.stats and args.stats.remove('mean')
    len(tasks) == 1 and 'median' in args.stats and args.stats.remove('median')
    extra_runs, extra_tasks = compute_stats(runs, args.stats, args.bins)
    runs += extra_runs
    tasks += extra_tasks

  print('Binning runs...', flush=True)
  if args.bins:
    maxs = collections.defaultdict(list)
    for run in runs:
      maxs[(run['task'], run['method'])].append(run['xs'].max())
    maxs = {k: max(vs) for k, vs in maxs.items()}
    for run in runs:
      if run['task'].startswith('stats_'):
        continue
      max_ = maxs[(run['task'], run['method'])] + 1e-8
      max_ = min(max_, args.xlim[1]) if args.xlim else max_
      step = max(1e-8, max_ / 30) if args.bins < 0 else args.bins
      borders = np.arange(0, max_, step)
      xs, ys = binning(run['xs'], run['ys'], borders, np.nanmean, fill='nan')
      run['xs'], run['ys'] = xs, ys

  print('Saving runs...', flush=True)
  filename = args.outdir / 'runs.json.gz'
  with gzip.open(filename, 'w') as f:
    f.write(json.dumps([
        {**run, 'xs': run['xs'].tolist(), 'ys': run['ys'].tolist()}
        for run in runs]).encode('utf-8'))
  console.print(f'Saved [green]{filename}[/green]')

  print('Plotting...', flush=True)
  fig, axes = plots(len(tasks), args.cols, args.size)
  for task, ax in zip(tasks, axes):
    title = TITLES.get(task, task.split('_', 1)[1].replace('_', ' ').title())
    ax.set_title(title)
    if not task.startswith('stats_'):
      args.xlim and ax.set_xlim(*args.xlim)
      args.ylim and ax.set_ylim(*args.ylim)
      args.xticks and ax.set_xticks(args.xticks)
    ax.xaxis.set_major_formatter(smart_format)
    # ax.tick_params(axis='both', labelsize=7)  # TOFO
  for task, ax in zip(tasks, axes):
    for i, method in enumerate(methods):
      relevant = [
          run for run in runs
          if run['task'] == task and run['method'] == method]
      if not relevant:
        console.print(f'Missing {method} on {task}!', style='red')
        continue
      if args.bins and args.agg:
        groups = [relevant]
      else:
        groups = [[run] for run in relevant]
      for group in groups:
        xs = group[0]['xs']
        ys = np.stack([run['ys'] for run in group], 0)
        mean = reduce(ys, np.nanmean, 0)
        std = reduce(ys, np.nanstd, 0)
        curve(
            ax, xs, mean, mean - std, mean + std,
            label=args.labels.get(method, method),
            order=i, color=args.colors(i))
  legendcols = args.legendcols or min(4, args.cols, len(axes))
  legend(fig, adjust=True, ncol=legendcols)
  if args.stats:
    for ax in axes[-len(extra_tasks):]:
      ax.set_facecolor((0.9, 0.9, 0.9))
  save(fig, args.outdir / 'curves.png')
  save(fig, args.outdir / 'curves.pdf')


def compute_stats(runs, stats, bins):
  extra_runs = []
  select = lambda baselines, name: {
      k: v[name] for k, v in baselines.items() if name in v}
  for stats in stats:
    if stats == 'tasks':
      extra_runs += stats_num_tasks(runs, bins)
    elif stats == 'mean':
      extra_runs += stats_self_norm(runs, bins, 'mean', np.nanmean)
    elif stats == 'median':
      extra_runs += stats_self_norm(runs, bins, 'median', np.nanmedian)
    elif stats == 'atari_mean':
      path = pathlib.Path('~/scores/atari_baselines.json').expanduser()
      baselines = json.loads(path.read_text())
      mins = select(baselines, 'random')
      maxs = select(baselines, 'human_gamer')
      extra_runs += stats_fixed_norm(
          runs, bins, mins, maxs, 'gamer_mean', np.nanmean)
    elif stats == 'atari_median':
      path = pathlib.Path('~/scores/atari_baselines.json').expanduser()
      baselines = json.loads(path.read_text())
      mins = select(baselines, 'random')
      maxs = select(baselines, 'human_gamer')
      extra_runs += stats_fixed_norm(
          runs, bins, mins, maxs, 'gamer_median', np.nanmedian)
    elif stats == 'atari_record':
      path = pathlib.Path('~/scores/atari_baselines.json').expanduser()
      baselines = json.loads(path.read_text())
      mins = select(baselines, 'random')
      maxs = select(baselines, 'human_record')
      extra_runs += stats_fixed_norm(
          runs, bins, mins, maxs, 'record_mean', np.nanmean)
    elif stats == 'atari_record_clip':
      path = pathlib.Path('~/scores/atari_baselines.json').expanduser()
      baselines = json.loads(path.read_text())
      mins = select(baselines, 'random')
      maxs = select(baselines, 'human_record')
      extra_runs += stats_fixed_norm(
          runs, bins, mins, maxs, 'record_mean_clip',
          lambda x, a: np.nanmean(np.minimum(x, 1), a))
    elif stats == 'dmlab_mean':
      path = pathlib.Path('~/scores/dmlab_baselines.json').expanduser()
      baselines = json.loads(path.read_text())
      mins = select(baselines, 'random')
      maxs = select(baselines, 'human')
      extra_runs += stats_fixed_norm(
          runs, bins, mins, maxs, 'human_mean',
          lambda vals, axis: np.nanmean(np.minimum(vals, 1), axis))
    else:
      raise NotImplementedError(stats)
  extra_tasks = natsort(set(run['task'] for run in extra_runs))
  return extra_runs, extra_tasks


def stats_self_norm(runs, bins, name='mean', aggregator=np.nanmean):
  methods = natsort(set(run['method'] for run in runs))
  seeds = natsort(set(run['seed'] for run in runs))
  lengths, mins, maxs = {}, {}, {}
  for run in runs:
    lengths[run['task']] = max(lengths.get(run['task'], 0), max(run['xs']))
    mins[run['task']] = min(mins.get(run['task'], np.inf), min(run['ys']))
    maxs[run['task']] = max(maxs.get(run['task'], -np.inf), max(run['ys']))
  if bins <= 0:
    borders = {
        task: np.linspace(0, length + 1e-8, 30)
        for task, length in lengths.items()}
  else:
    border = np.arange(0, max(lengths.values()) + 1e-8, bins)
    borders = {task: border for task, length in lengths.items()}
  extra_runs = []
  for method in methods:
    for seed in seeds:
      scores = []
      for run in runs:
        if not (run['method'] == method and run['seed'] == seed):
          continue
        task = run['task']
        if np.isclose(mins[task], maxs[task]):
          continue
        _, ys = binning(
            run['xs'], run['ys'], borders[task], np.nanmean, fill='last')
        scores.append((ys - mins[task]) / (maxs[task] - mins[task]))
      if scores:
        scores = np.array(scores)
        xs = np.linspace(0, 1, len(scores[0]))
        extra_runs.append({
            'task': f'stats_normalized_{name}', 'method': method, 'seed': seed,
            'xs': xs, 'ys': reduce(scores, aggregator, 0)})
  return extra_runs


def stats_fixed_norm(
    runs, bins, mins, maxs, name='mean', aggregator=np.nanmean):
  methods = natsort(set(run['method'] for run in runs))
  seeds = natsort(set(run['seed'] for run in runs))
  lengths = {}
  for run in runs:
    lengths[run['task']] = max(lengths.get(run['task'], 0), max(run['xs']))
  if bins <= 0:
    borders = {
        task: np.linspace(0, length + 1e-8, 30)
        for task, length in lengths.items()}
  else:
    border = np.arange(0, max(lengths.values()) + 1e-8, bins)
    borders = {task: border for task, length in lengths.items()}
  extra_runs = []
  for method in methods:
    for seed in seeds:
      scores = []
      for run in runs:
        if not (run['method'] == method and run['seed'] == seed):
          continue
        task = run['task']
        _, ys = binning(
            run['xs'], run['ys'], borders[task], np.nanmean, fill='last')
        if task == 'atari_jamesbond' and 'atari_james_bond' in mins:
          task = 'atari_james_bond'
        scores.append((ys - mins[task]) / (maxs[task] - mins[task]))
      if scores:
        xs = np.linspace(0, 1, len(scores[0]))
        extra_runs.append({
            'task': f'stats_{name}', 'method': method, 'seed': seed,
            'xs': xs, 'ys': reduce(scores, aggregator, 0)})
  return extra_runs


def stats_num_tasks(runs, bins):
  methods = natsort(set(run['method'] for run in runs))
  seeds = natsort(set(run['seed'] for run in runs))
  lengths = {}
  for run in runs:
    lengths[run['task']] = max(lengths.get(run['task'], 0), max(run['xs']))
  if bins <= 0:
    borders = {
        task: np.linspace(0, length + 1e-8, 30)
        for task, length in lengths.items()}
  else:
    border = np.arange(0, max(lengths.values()) + 1e-8, bins)
    borders = {task: border for task, length in lengths.items()}
  extra_runs = []
  for method in methods:
    for seed in seeds:
      nonempty = []
      for run in runs:
        if not (run['method'] == method and run['seed'] == seed):
          continue
        task = run['task']
        _, ys = binning(
            run['xs'], run['ys'], borders[task], np.nanmean, fill='nan')
        nonempty.append(np.isfinite(ys))
      if nonempty:
        xs = np.linspace(0, 1, len(nonempty[0]))
        extra_runs.append({
            'task': 'stats_number_of_tasks', 'method': method, 'seed': seed,
            'xs': xs, 'ys': np.sum(nonempty, 0)})
  return extra_runs


def load_metrics(
    directory, pattern, xaxis, yaxis, yaxis2, seed_prefix=None,
    method_prefix=None, tasks=(r'.*',), methods=(r'.*',), workers=1):
  console = rich.console.Console()
  directory = directory.expanduser().resolve()
  tasks = [re.compile(regex) for regex in tasks]
  methods = [re.compile(regex) for regex in methods]
  runs = []
  for filename in directory.glob(pattern):
    task, method, seed = filename.parts[-4:-1]
    if not any(p.search(task) for p in tasks):
      continue
    if not any(p.search(method) for p in methods):
      continue
    if seed_prefix:
      seed = f'{seed_prefix}_{seed}'
    if method_prefix:
      method = f'{method_prefix}_{method}'
    runs.append({
        'task': task, 'method': method, 'seed': seed, 'filename': filename})
  console.print(f'Loading {len(runs)} runs from [green]{directory}[/green]...')
  jobs = [
      functools.partial(load_run, run, xaxis, yaxis, yaxis2) for run in runs]
  if workers > 1:
    with mp.Pool(workers) as pool:
      promises = [pool.apply_async(j) for j in jobs]
      runs = [promise.get() for promise in tqdm.tqdm(promises)]
  else:
    runs = [job() for job in tqdm.tqdm(jobs)]
  runs = [r for r in runs if r is not None]
  return runs


def load_run(run, xaxis, yaxis, yaxis2):
  try:
    console = rich.console.Console()
    filename = run.pop('filename')
    try:
      df = pd.read_json(filename, lines=True)
    except ValueError:
      records = []
      for i, line in enumerate(pathlib.Path(filename).read_text().split('\n')):
        if not line:
          continue
        try:
          records.append(json.loads(line))
        except ValueError:
          print(f'Skipping invalid JSON line {i} in {filename}.')
      df = pd.DataFrame(records)
    yaxis = yaxis if yaxis in df.columns else yaxis2
    df = df[[xaxis, yaxis]].dropna()
    run['xs'] = df[xaxis].to_numpy()
    run['ys'] = df[yaxis].to_numpy()
    return run
  except Exception as e:
    console.print(
        f'Exception loading {run["method"]} on {run["task"]}:\n {e}',
        style='red')
    return None


def plots(
    amount, cols=4, size=(2, 2.3), xticks=4, yticks=5, grid=(1, 1), **kwargs):
  cols = min(cols, amount)
  rows = int(np.ceil(amount / cols))
  size = (cols * size[0], rows * size[1])
  fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, **kwargs)
  axes = axes.flatten()
  for ax in axes:
    ax.xaxis.set_major_locator(ticker.MaxNLocator(xticks))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(yticks))
    if grid:
      grid = (grid, grid) if not hasattr(grid, '__len__') else grid
      ax.grid(which='both', color='#eeeeee')
      ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[0])))
      ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(int(grid[1])))
      ax.tick_params(which='minor', length=0)
  for ax in axes[amount:]:
    ax.axis('off')
  axes = axes[:amount]
  return fig, axes


def curve(ax, xs, ys, low=None, high=None, label=None, order=0, **kwargs):
  finite = np.isfinite(ys)
  ax.plot(
      xs[finite], ys[finite],
      label=label, zorder=1000 - order, **kwargs)
  if low is not None and finite.sum() > 1:
    ax.fill_between(
        xs[finite], low[finite], high[finite],
        zorder=100 - order, alpha=0.2, lw=0, **kwargs)


def legend(fig, mapping=None, adjust=False, **kwargs):
  options = dict(
      fontsize='medium', numpoints=1, labelspacing=0, columnspacing=1.2,
      handlelength=1.5, handletextpad=0.5, ncol=4, loc='lower center')
  options.update(kwargs)
  entries = {}
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      if mapping and label in mapping:
        label = mapping[label]
      entries[label] = handle
  leg = fig.legend(entries.values(), entries.keys(), **options)
  leg.get_frame().set_edgecolor('white')
  if adjust is not False:
    pad = adjust if isinstance(adjust, (int, float)) else 0.5
    extent = leg.get_window_extent(fig.canvas.get_renderer())
    extent = extent.transformed(fig.transFigure.inverted())
    yloc, xloc = options['loc'].split()
    y0 = dict(lower=extent.y1, center=0, upper=0)[yloc]
    y1 = dict(lower=1, center=1, upper=extent.y0)[yloc]
    x0 = dict(left=extent.x1, center=0, right=0)[xloc]
    x1 = dict(left=1, center=1, right=extent.x0)[xloc]
    fig.tight_layout(rect=[x0, y0, x1, y1], h_pad=pad, w_pad=pad)


def smart_format(x, pos=None):
  if abs(x) < 1e3:
    if float(int(x)) == float(x):
      return str(int(x))
    return str(round(x, 10)).rstrip('0')
  if abs(x) < 1e6:
    return f'{x/1e3:.0f}K' if x == x // 1e3 * 1e3 else f'{x/1e3:.1f}K'
  if abs(x) < 1e9:
    return f'{x/1e6:.0f}M' if x == x // 1e6 * 1e6 else f'{x/1e6:.1f}M'
  return f'{x/1e9:.0f}B' if x == x // 1e9 * 1e9 else f'{x/1e9:.1f}B'


def save(fig, filename):
  console = rich.console.Console()
  filename = pathlib.Path(filename).expanduser()
  filename.parent.mkdir(parents=True, exist_ok=True)
  fig.savefig(filename)
  console.print(f'Saved [green]{filename}[/green]')
  if filename.suffix == '.pdf':
    try:
      subprocess.call(['pdfcrop', str(filename), str(filename)])
    except FileNotFoundError:
      print('Install LaTeX to crop PDF outputs.')


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  xs = xs if isinstance(xs, np.ndarray) else np.array(xs)
  ys = ys if isinstance(ys, np.ndarray) else np.array(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  for start, stop in zip(borders[:-1], borders[1:]):
    left = (xs <= start).sum()
    right = (xs <= stop).sum()
    if left < right:
      value = reduce(ys[left:right], reducer)
    elif binned:
      value = {'nan': np.nan, 'last': binned[-1]}[fill]
    else:
      value = np.nan
    binned.append(value)
  return borders[1:], np.array(binned)


def reduce(values, reducer=np.nanmean, *args, **kwargs):
  with warnings.catch_warnings():  # Buckets can be empty.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    return reducer(values, *args, **kwargs)


def natsort(sequence):
  pattern = re.compile(r'([0-9]+)')
  return sorted(sequence, key=lambda x: [
      (int(y) if y.isdigit() else y) for y in pattern.split(x)])


def parse_args(argv=None):
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--indirs', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--outdir', type=pathlib.Path, required=True)
  parser.add_argument('--pattern', type=str, default='**/scores.jsonl')
  parser.add_argument('--prefix', type=boolean, default=False)
  parser.add_argument('--xaxis', type=str, default='step')
  parser.add_argument('--yaxis', type=str, default='episode/score')
  parser.add_argument('--yaxis2', type=str, default='eval_episode/score')
  parser.add_argument('--tasks', nargs='+', default=[r'.*'])
  parser.add_argument('--methods', nargs='+', default=[r'.*'])
  parser.add_argument('--bins', type=float, default=-1)
  parser.add_argument('--agg', type=boolean, default=True)
  parser.add_argument('--size', nargs=2, type=float, default=[2.5, 2.3])
  parser.add_argument('--cols', type=int, default=6)
  parser.add_argument('--legendcols', type=int, default=0)
  parser.add_argument('--xlim', nargs=2, type=float, default=None)
  parser.add_argument('--ylim', nargs=2, type=float, default=None)
  parser.add_argument('--xticks', nargs='+', type=float, default=None)
  parser.add_argument('--labels', nargs='+', default=[])
  parser.add_argument('--colors', type=str, nargs='+', default=['contrast'])
  parser.add_argument('--workers', type=int, default=12)
  parser.add_argument('--stats', type=str, nargs='*', default=[
      'mean', 'median', 'tasks'])
  args = parser.parse_args(argv)
  args.indirs = tuple([x.expanduser() for x in args.indirs])
  args.outdir = args.outdir.expanduser() / args.indirs[0].stem
  assert len(args.labels) % 2 == 0
  args.labels = {k: v for k, v in zip(args.labels[:-1], args.labels[1:])}
  if len(args.colors) == 1:
    try:
      args.colors = plt.get_cmap(args.colors[0])
    except ValueError:
      if args.colors[0] in COLORS:
        cmap = COLORS[args.colors[0]]
      else:
        cmap = args.colors
      args.colors = lambda i: cmap[i % len(cmap)]
  if args.stats == ['none']:
    args.stats = []
  return args


if __name__ == '__main__':
  main()
