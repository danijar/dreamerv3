import argparse
import functools
import json
import multiprocessing as mp
import pathlib
import re

import numpy as np
import pandas as pd
import rich.console
import tqdm


def main():
  console = rich.console.Console()
  args = parse_args()
  paths = []
  for directory in args.indirs:
    paths += list(directory.expanduser().resolve().glob(args.pattern))
  tensor, tasks, methods, seeds = load_scores(sorted(set(paths)), args)
  console.print(f'Tasks ({len(tasks)}): [cyan]{", ".join(tasks)}[/cyan]')
  console.print(f'Methods ({len(methods)}): [cyan]{", ".join(methods)}[/cyan]')
  console.print(f'Seed ({len(seeds)}): [cyan]{", ".join(seeds)}[/cyan]')
  if not tasks or not methods or not seeds:
    console.print('Nothing to print!', style='red')
    return

  path = pathlib.Path('~/scores/atari_baselines.json').expanduser()
  baselines = json.loads(path.read_text())
  select = lambda baselines, name: {
      k: v[name] for k, v in baselines.items() if name in v}
  if args.normalize:
    mins = select(baselines, 'random')
    maxs = select(baselines, 'human_gamer')
    mins = np.array([mins[task] for task in tasks])
    maxs = np.array([maxs[task] for task in tasks])

  # maxs = maxs[:, None, None]
  # mins = mins[:, None, None]
  # normed = (tensor - mins) / (maxs - mins)
  # means = 100 * np.nanmean(normed, 0)
  # medians = 100 * np.nanmedian(normed, 0)

  averaged = np.round(np.nanmean(tensor, -1), 3)
  if args.normalize:
    maxs = maxs[:, None]
    mins = mins[:, None]
    normed = (averaged - mins) / (maxs - mins)
  else:
    normed = averaged
  means = 100 * np.nanmean(normed, 0)
  medians = 100 * np.nanmedian(normed, 0)

  completed = np.isfinite(normed).sum(0)
  print('')
  print('Methods:', methods)
  print('Means:', means)
  print('Medians:', medians)
  print('Tasks:', tasks)
  for i, method in enumerate(methods):
    # raw = np.round(np.nanmean(tensor[:, i], -1), 3).tolist()
    print('\n', {method: dict(zip(tasks, averaged[:, i]))})
  for i, method in enumerate(methods):
    mean = means[i]
    median = medians[i]
    print(f'\n{method}')
    # print(f' Mean HNS:     {np.nanmean(mean):6.0f} ±{np.nanstd(mean):.0f}')
    # print(f' Median HNS:   {np.nanmean(median):6.0f} ±{np.nanstd(median):.0f}')
    print(f' Mean HNS:    {mean:.1f}')
    print(f' Median HNS:  {median:.1f}')
    print(f' Completed:   {completed[i]}')


def load_scores(paths, args):
  console = rich.console.Console()
  tasks, methods, seeds = zip(*[x.parts[-4:-1] for x in paths])
  matched = lambda name, patterns: any(re.search(p, name) for p in patterns)
  tasks = [x for x in natsort(set(tasks)) if matched(x, args.tasks)]
  methods = [x for x in natsort(set(methods)) if matched(x, args.methods)]
  seeds = natsort(set(seeds))
  paths = [x for x in paths if x.parts[-4] in tasks and x.parts[-3] in methods]
  console.print(f'Loading {len(paths)} scores...')
  jobs = [functools.partial(load_score, path, args) for path in paths]
  if args.workers > 1:
    with mp.Pool(args.workers) as pool:
      promises = [pool.apply_async(j) for j in jobs]
      scores = [promise.get() for promise in tqdm.tqdm(promises)]
  else:
    scores = [job() for job in tqdm.tqdm(jobs)]
  tensor = np.empty((len(tasks), len(methods), len(seeds)))
  tensor[:] = np.nan
  for path, score in zip(paths, scores):
    if score is None:
      pass
    task, method, seed = path.parts[-4:-1]
    tensor[tasks.index(task), methods.index(method), seeds.index(seed)] = score
  return tensor, tasks, methods, seeds


def load_score(path, args):
  try:
    console = rich.console.Console()
    task, method, seed = path.parts[-4:-1]
    df = load_json(path)
    df = df[[args.xaxis, args.yaxis]].dropna()
    xs = df[args.xaxis].to_numpy()
    ys = df[args.yaxis].to_numpy()
    if not len(xs):
      console.print(
          f'Skipping {task} {method} {seed} that has not reported scores!',
          style='red')
      return None
    # if xs[-1] < args.point - args.before:
    #   console.print(
    #       f'Skipping {task} {method} {seed} that only reached to step '
    #       f'{xs[-1]} but needed {args.point - args.before}!', style='red')
    #   return None

    # stop = (xs <= args.point + args.tolerance).sum()
    # start = (xs < args.point + args.tolerance - args.before).sum()
    # start = min(start, start - 1)

    stop = (xs <= args.point + args.tolerance).sum() + 2
    start = max(0, stop - args.episodes)

    assert start < stop, (start, stop, task, method, seed, xs)
    score = ys[start: stop].mean()
    return score
  except Exception as e:
    console.print(f'Exception loading {path}:\n {e}', style='red')
    return None


def load_json(path):
  try:
    return pd.read_json(path, lines=True)
  except ValueError:
    records = []
    for i, line in enumerate(pathlib.Path(path).read_text().split('\n')):
      if not line:
        continue
      try:
        records.append(json.loads(line))
      except ValueError:
        print(f'Skipping invalid JSON line {i} in {path}.')
    return pd.DataFrame(records)


def natsort(sequence):
  pattern = re.compile(r'([0-9]+)')
  return sorted(sequence, key=lambda x: [
      (int(y) if y.isdigit() else y) for y in pattern.split(x)])


def parse_args(argv=None):
  boolean = lambda x: bool(['False', 'True'].index(x))
  parser = argparse.ArgumentParser()
  parser.add_argument('--indirs', nargs='+', type=pathlib.Path, required=True)
  parser.add_argument('--pattern', type=str, default='**/scores.jsonl')
  parser.add_argument('--workers', type=int, default=1)
  parser.add_argument('--tasks', nargs='+', default=[r'.*'])
  parser.add_argument('--methods', nargs='+', default=[r'.*'])
  parser.add_argument('--xaxis', type=str, default='step')
  parser.add_argument('--yaxis', type=str, default='episode/score')
  parser.add_argument('--point', type=float, default=4e5)
  parser.add_argument('--before', type=float, default=3e4)
  parser.add_argument('--tolerance', type=float, default=100)
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--normalize', type=boolean, default=True)
  parser.add_argument('--stats', type=str, nargs='*', default=[
      'mean', 'median', 'tasks'])
  args = parser.parse_args(argv)
  args.indirs = tuple([x.expanduser() for x in args.indirs])
  if args.stats == ['none']:
    args.stats = []
  return args


if __name__ == '__main__':
  main()
