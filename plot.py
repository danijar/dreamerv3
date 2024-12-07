import concurrent.futures
import functools
import json
import re
import warnings

import elements
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ruamel.yaml as yaml
import tqdm


COLORS = [
    '#0022ff', '#33aa00', '#ff0011', '#ddaa00', '#cc44dd', '#0088aa',
    '#001177', '#117700', '#990022', '#885500', '#553366', '#006666',
    '#7777cc', '#999999', '#990099', '#888800', '#ff00aa', '#444444',
]


def load_run(filename, xkeys, ykeys, ythres=None):
  try:
    try:
      df = pd.read_json(filename, lines=True)
    except ValueError:
      print('Falling back to robust JSONL reader.')
      records = []
      for line in filename.read_text().split('\n')[:-1]:
        try:
          records.append(json.loads(line))
        except json.decoder.JSONDecodeError:
          print(f'Skipping invalid JSONL line: {line}')
      df = pd.DataFrame(records)
    assert len(df), 'no timesteps in run'
    xkey = [k for k in xkeys if k in df]
    ykey = [k for k in ykeys if k in df]
    assert xkey, (filename, df.columns, xkeys)
    assert ykey, (filename, df.columns, ykeys)
    xs = df[xkey[0]].to_list()
    ys = df[ykey[0]].to_list()
    assert isinstance(xs, list), type(xs)
    assert isinstance(ys, list), type(ys)
    if ythres:
      ys = [1 if y > ythres else 0 for y in ys]
    return xs, ys
  except Exception as e:
    elements.print(f'Exception loading {filename}: {e}', color='red')
    return None


def load_runs(args):
  indirs = [elements.Path(x) for x in args.indirs]
  assert len(set(x.name for x in indirs)) == len(indirs), indirs
  records, filenames = [], []
  methods = re.compile(args.methods)
  tasks = re.compile(args.tasks)
  for indir in indirs:
    found = list(indir.glob(args.pattern))
    assert found, (indir, args.pattern)
    for filename in found:
      if args.newstyle:
        _, task, method, seed = filename.parent.name.split('-')
      else:
        task, method, seed = str(filename).split('/')[-4: -1]
      if not (methods.search(method) and tasks.search(task)):
        continue
      seed = f'{indir.name}_{seed}' if len(args.indirs) > 1 else seed
      method = f'{indir.name}_{method}' if args.indir_prefix else method
      records.append(dict(task=task, method=method, seed=seed))
      filenames.append(filename)
  print(f'Loading {len(records)} runs...')
  load = functools.partial(
      load_run, xkeys=args.xkeys, ykeys=args.ykeys, ythres=args.ythres)
  if args.workers:
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
      runs = list(tqdm.tqdm(pool.map(load, filenames), total=len(filenames)))
  else:
    runs = list(tqdm.tqdm((load(x) for x in filenames), total=len(filenames)))
  assert len(runs) > 0
  records, runs = zip(*[(x, y) for x, y in zip(records, runs) if y])
  for record, (xs, ys) in zip(records, runs):
    record.update(xs=xs, ys=ys)
  return pd.DataFrame(records)


def bin_runs(df, args):
  print('Binning runs...')
  if args.xlim:
    df['xlim'] = args.xlim
  else:
    xlim = df.groupby('task')['xs'].agg(lambda xs: max(max(x) for x in xs))
    df = pd.merge(df, xlim.rename('xlim'), on='task', how='left')
  if args.binsize:
    df['xlim'] = df['xlim'].max()
    df['binsize'] = args.binsize
  else:
    assert args.bins <= 1000, args.bins
    df['binsize'] = df['xlim'].apply(lambda x: x / args.bins)
  def binning(row):
    bins = np.arange(0, row['xlim'] + 0.99 * row['binsize'], row['binsize'])
    sums = np.histogram(row['xs'], bins=bins, weights=row['ys'])[0]
    nums = np.histogram(row['xs'], bins=bins)[0]
    xs = bins[1:]
    ys = np.divide(sums, nums, out=np.full(len(xs), np.nan), where=(nums != 0))
    return xs, ys
  df['xs'], df['ys'] = zip(*df.apply(binning, axis=1))
  df = df.drop(columns=['xlim', 'binsize'])
  assert len(df['xs'].apply(len).unique()) == 1
  return df


def comp_stat(name, df, fn, baseline=None):
  df = df.copy()
  if not df['xs'].apply(lambda xs: np.array_equal(xs, df['xs'][0])).all():
    assert len(df['xs'].apply(len).unique()) == 1
    domain = np.linspace(0, 1, len(df['xs'][0]))
    df['xs'] = df['xs'].apply(lambda _: domain)

  df = df.groupby(['task', 'method'])[['xs', 'ys']].agg(np.stack).reset_index()
  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(lambda ys: nanmean(ys, axis=0))
  if baseline is not None:
    def normalize(row):
      lo, hi = baseline[row['task']]
      return (row['ys'] - lo) / (hi - lo)
    df['ys'] = df.apply(normalize, axis=1)
  df = df.groupby('method')[['xs', 'ys']].agg(np.stack).reset_index()

  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(fn)
  df['name'] = name
  return df


def comp_count(name, df):
  df = df.copy()
  if not df['xs'].apply(lambda xs: np.array_equal(xs, df['xs'][0])).all():
    assert len(df['xs'].apply(len).unique()) == 1
    domain = np.linspace(0, 1, len(df['xs'][0]))
    df['xs'] = df['xs'].apply(lambda _: domain)
  df = df.groupby(['method'])[['xs', 'ys']].agg(np.stack).reset_index()
  df['xs'] = df['xs'].apply(lambda xs: nanmean(xs, axis=0))
  df['ys'] = df['ys'].apply(lambda ys: np.isfinite(ys).sum(0))
  df['name'] = name
  return df


def comp_stats(df, args):
  print('Computing stats...')
  refs = yaml.YAML(typ='safe').load(
      (elements.Path(__file__).parent / 'baselines.yaml').read())
  self_baseline = df.groupby('task')['ys'].agg(lambda ys: (
      min(min(y) for y in ys), max(max(y) for y in ys))).to_dict()
  stats = []

  choices = list(args.stats)
  choices = [x for x in choices if x != 'none']
  if not choices:
    return None

  if 'auto' in choices:
    choices.remove('auto')
    if all(x.startswith('atari_') for x in df.task.unique()):
      choices += ['atari_mean', 'atari_median']
    if all(x.startswith('dmc_') for x in df.task.unique()):
      choices += ['mean', 'median']
    if all(x.startswith('dmlab_') for x in df.task.unique()):
      choices += ['dmlab_mean', 'dmlab_mean_capped']
    if all(x.startswith('procgen_') for x in df.task.unique()):
      choices += ['procgen_mean']

  ax0 = lambda fn: functools.partial(fn, axis=0)
  for stat in choices:
    if stat == 'runs':
      x = comp_count('Runs', df)
    elif stat == 'mean':
      x = comp_stat('Mean', df, ax0(np.mean))
    elif stat == 'median':
      x = comp_stat('Median', df, ax0(np.median))
    elif stat == 'self_mean':
      x = comp_stat('Self Mean', df, ax0(nanmean), self_baseline)
    elif stat == 'self_median':
      x = comp_stat('Self Median', df, ax0(nanmedian), self_baseline)
    elif stat == 'atari_mean':
      x = comp_stat('Gamer Mean', df, ax0(np.mean), refs['atari57_gamer'])
    elif stat == 'atari_median':
      x = comp_stat('Gamer Median', df, ax0(np.median), refs['atari57_gamer'])
    elif stat == 'dmlab_mean':
      x = comp_stat('Capped Mean', df, ax0(np.mean), refs['dmlab30'])
    elif stat == 'dmlab_mean_capped':
      fn = lambda x: np.minimum(x, 1).mean(0)
      x = comp_stat('Capped Mean', df, fn, refs['dmlab30'])
    elif stat == 'procgen_mean':
      x = comp_stat('Normalized Mean', df, ax0(np.mean), refs['procgen_hard'])
    else:
      raise ValueError(stat)
    stats.append(x)
  return pd.concat(stats)


def plot_runs(df, stats, args):
  print('Plotting...')
  tasks = natsort(df.task.unique())
  snames = [] if stats is None else stats.name.unique()
  methods = natsort(df.method.unique())
  total = len(tasks) + len(snames)
  cols = args.cols or (4 + (total > 24) + (total > 35) + (total > 48))
  fig, axes = plots(total, cols, args.size)

  grouped = df.groupby(['task', 'method'])[['xs', 'ys', 'seed']].agg(np.stack)
  for task, ax in zip(tasks, axes[:len(tasks)]):
    style(ax, xticks=args.xticks, yticks=args.yticks)
    title = task.replace('_', ' ').replace(':', ' ').split(' ', 1)[1].title()
    ax.set_title(title)
    args.xlim and ax.set_xlim(0, 1.03 * args.xlim)
    args.ylim and ax.set_ylim(0, 1.03 * args.ylim)
    for i, method in enumerate(methods):
      try:
        sub = grouped.loc[task, method]
      except KeyError:
        print(f"Missing method '{method}' on task '{task}'")
        continue
      bins = sub['xs'][0]

      if args.agg:
        mean = nanmean(sub['ys'], 0)
        std = nanstd(sub['ys'], 0)
        curve(ax, bins, mean, mean - std, mean + std, method, i)
      else:
        for j in range(sub['xs'].shape[0]):
          curve(ax, sub['xs'][j], sub['ys'][j], None, None, method, i)

  if stats is not None:
    grouped = stats.groupby(['name', 'method'])[['xs', 'ys']].agg(np.stack)
    for sname, ax in zip(snames, axes[len(tasks):]):
      style(ax, xticks=args.xticks, yticks=args.yticks, darker=True)
      ax.set_title(sname)
      args.xlim and ax.set_xlim(0, 1.03 * args.xlim)
      for i, method in enumerate(methods):
        sub = grouped.loc[sname, method]
        curve(ax, sub['xs'], sub['ys'], None, None, method, i)

  legend(fig, adjust=True, ncol=args.legendcols or min(4, cols, len(axes)))

  outdir = elements.Path(args.outdir) / elements.Path(args.indirs[0]).stem
  outdir.mkdir()
  filename = outdir / 'curves.png'
  fig.savefig(filename, dpi=150)
  print('Saved', filename)


def plots(amount, cols=4, size=(3, 3), **kwargs):
  rows = int(np.ceil(amount / cols))
  cols = min(cols, amount)
  kwargs['figsize'] = kwargs.get('figsize', (size[0] * cols, size[1] * rows))
  fig, axes = plt.subplots(nrows=rows, ncols=cols, squeeze=False, **kwargs)
  for ax in axes.flatten()[amount:]:
    ax.axis('off')
  ax = axes.flatten()[:amount]
  return fig, ax


def style(ax, xticks=4, yticks=4, grid=(1, 1), logx=False, darker=False):
  ax.tick_params(axis='x', which='major', length=2, labelsize=10, pad=3)
  ax.tick_params(axis='y', which='major', length=2, labelsize=10, pad=2)
  ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(xticks))
  ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(yticks))
  ax.xaxis.set_major_formatter(lambda x, pos: natfmt(x))
  ax.yaxis.set_major_formatter(lambda x, pos: natfmt(x))
  if grid:
    color = '#cccccc' if darker else '#eeeeee'
    ax.grid(which='both', color=color)
    ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid[0]))
    ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(grid[1]))
    ax.tick_params(which='minor', length=0)
  if logx:
    ax.set_xscale('log')
    ax.xaxis.set_major_locator(plt.LogLocator(10, numticks=3))
    ax.xaxis.set_minor_locator(plt.LogLocator(10, subs='all', numticks=100))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
  if darker:
    ax.set_facecolor((0.95, 0.95, 0.95))


def curve(
    ax, xs, ys, lo=None, hi=None, label=None, order=None, color=None,
    scatter=True, **kwargs):
  color = color or (None if order is None else COLORS[order])
  order = order or 0
  kwargs['color'] = color
  mask = np.isfinite(ys)
  ax.plot(xs[mask], ys[mask], label=label, zorder=200 - order, **kwargs)
  if scatter:
    ax.scatter(xs, ys, s=5, label=label, zorder=3000 - order, **kwargs)
  if lo is not None:
    ax.fill_between(
        xs[mask], lo[mask], hi[mask],
        zorder=100 - order, lw=0, **{**kwargs, 'alpha': 0.2})


def legend(fig, names=None, reverse=False, adjust=False, **kwargs):
  options = dict(
      fontsize=10, numpoints=1, labelspacing=0, columnspacing=1.2,
      handlelength=1.5, handletextpad=0.5, ncol=4, loc='lower center')
  options.update(kwargs)
  entries = {}
  for ax in fig.axes:
    for handle, label in zip(*ax.get_legend_handles_labels()):
      entries[label] = handle
  if names:
    entries = {name: entries[label] for label, name in names.items()}
  if reverse:
    entries = dict(list(reversed(list(entries.items()))))
  leg = fig.legend(entries.values(), entries.keys(), **options)
  leg.get_frame().set_edgecolor('white')
  leg.set_zorder(2000)
  [line.set_linewidth(2) for line in leg.legend_handles]
  if adjust:
    extent = leg.get_window_extent(fig.canvas.get_renderer())
    extent = extent.transformed(fig.transFigure.inverted())
    yloc, xloc = options['loc'].split()
    y0 = dict(lower=extent.y1, center=0, upper=0)[yloc]
    y1 = dict(lower=1, center=1, upper=extent.y0)[yloc]
    x0 = dict(left=extent.x1, center=0, right=0)[xloc]
    x1 = dict(left=1, center=1, right=extent.x0)[xloc]
    fig.tight_layout(rect=[x0, y0, x1, y1], h_pad=1, w_pad=1)
  return leg


def silent(fn):
  def wrapped(*args, **kwargs):
    with warnings.catch_warnings():
      warnings.simplefilter('ignore')
      return fn(*args, **kwargs)
  return wrapped
nanmean = silent(np.nanmean)
nanmedian = silent(np.nanmedian)
nanstd = silent(np.nanstd)
nanmax = silent(np.nanmax)
nanmin = silent(np.nanmin)


def natsort(sequence):
  pattern = re.compile(r'([0-9]+)')
  return sorted(sequence, key=lambda x: [
      (int(y) if y.isdigit() else y) for y in pattern.split(x)])


def natfmt(x):

  if abs(x) < 1e3:
    x, suffix = x, ''
  elif 1e3 <= abs(x) < 1e6:
    x, suffix = x / 1e3, 'K'
  elif 1e6 <= abs(x) < 1e9:
    x, suffix = x / 1e6, 'M'
  elif 1e9 <= abs(x):
    x, suffix = x / 1e9, 'B'
  if abs(x) <= 1:
    return f'{x:.3f}{suffix}'
  elif 1 <= abs(x) < 10:
    return f'{x:.1f}{suffix}'
  elif 10 <= abs(x):
    return f'{x:.0f}{suffix}'


def print_summary(df):
  methods = natsort(df.method.unique())
  tasks = natsort(df.task.unique())
  seeds = natsort(df.seed.unique())
  print('-' * 79)
  print(f'Methods ({len(methods)}):', ', '.join(methods))
  print('-' * 79)
  print(f'Tasks ({len(tasks)}):', ', '.join(tasks))
  print('-' * 79)
  print(f'Seeds ({len(seeds)}):', ', '.join(seeds))
  print('-' * 79)


def main(args):
  df = load_runs(args)
  df = bin_runs(df, args)
  print_summary(df)
  if args.todf:
    assert args.todf.endswith('.json.gz')
    import ipdb; ipdb.set_trace()
    df.to_json(args.todf, orient='records')
    print(f'Saved {args.todf}')
  stats = comp_stats(df, args)
  plot_runs(df, stats, args)


if __name__ == '__main__':
  main(elements.Flags(
      pattern='**/scores.jsonl',
      indirs=[''],
      outdir='',
      methods='.*',
      tasks='.*',
      newstyle=True,
      indir_prefix=False,
      workers=16,
      xkeys=['xs', 'step'],
      ykeys=['ys', 'episode/score'],
      ythres=0.0,
      xlim=0,
      ylim=0,
      binsize=0,
      bins=30,
      cols=0,
      legendcols=0,
      size=[3, 3],
      xticks=4,
      yticks=10,
      stats=['runs', 'auto'],
      agg=True,
      todf='',
  ).parse())
