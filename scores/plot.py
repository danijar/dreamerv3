import gzip
import json
import pathlib
import re
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


COLORS = (
    '#377eb8', '#4daf4a', '#984ea3', '#e41a1c', '#ff7f00', '#a65628',
    '#f781bf', '#888888', '#a6cee3', '#b2df8a', '#cab2d6', '#fb9a99',
)


def plots(
    amount, cols=4, size=(2, 2.3), xticks=4, yticks=5, grid=(1, 1), **kwargs):
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
  return fig, axes


def curve(
    ax, domain, values, low=None, high=None, label=None, order=0, **kwargs):
  finite = np.isfinite(values)
  ax.plot(
      domain[finite], values[finite],
      label=label, zorder=1000 - order, **kwargs)
  if low is not None:
    ax.fill_between(
        domain[finite], low[finite], high[finite],
        zorder=100 - order, alpha=0.2, lw=0, **kwargs)


def legend(fig, mapping=None, adjust=False, **kwargs):
  options = dict(
      fontsize='medium', numpoints=1, labelspacing=0, columnspacing=1.2,
      handlelength=1.5, handletextpad=0.5, ncol=4, loc='lower center')
  options.update(kwargs)
  # Find all labels and remove duplicates.
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


def binning(xs, ys, borders, reducer=np.nanmean, fill='nan'):
  assert fill in ('nan', 'last', 'zeros')
  xs = xs if isinstance(xs, np.ndarray) else np.asarray(xs)
  ys = ys if isinstance(ys, np.ndarray) else np.asarray(ys)
  order = np.argsort(xs)
  xs, ys = xs[order], ys[order]
  binned = []
  for start, stop in zip(borders[:-1], borders[1:]):
    left = (xs <= start).sum()
    right = (xs <= stop).sum()
    value = np.nan
    if left < right:
      value = reduce(ys[left:right], reducer)
    if np.isnan(value):
      if fill == 'zeros':
        value = 0
      if fill == 'last' and binned:
        value = binned[-1]
    binned.append(value)
  return borders[1:], np.array(binned)


def reduce(values, reducer=np.nanmean, *args, **kwargs):
  with warnings.catch_warnings():  # Buckets can be empty.
    warnings.simplefilter('ignore', category=RuntimeWarning)
    return reducer(values, *args, **kwargs)


datadir = pathlib.Path(__file__).parent / 'data'
outdir = pathlib.Path(__file__).parent / 'figs'
outdir.mkdir(exist_ok=True)

suites = sorted(set(x.name.split('_')[0] for x in datadir.glob('*.json.gz')))

if len(sys.argv) > 1:
  suites = [x for x in suites if re.search(sys.argv[1], x)]
  print(f'Pattern matches {len(suites)} suites: {", ".join(suites)}')

for suite in suites:
  print('-' * 79)
  print(suite)
  print('-' * 79)

  runs = []
  for filename in datadir.glob(f'{suite}_*.json.gz'):
    with gzip.open(filename, 'rb') as f:
      runs += json.load(f)

  tasks = sorted(set(run['task'] for run in runs))
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))

  fig, axes = plots(len(tasks), cols=6, size=(2, 2))
  for i, task in enumerate(tasks):
    ax = axes[i]

    title = task.split('_', 1)[-1]
    title = title.replace('_', ' ').title()
    ax.set_title(title)

    for j, method in enumerate(methods):
      relevant = [run for run in runs if (
          run['task'] == task and run['method'] == method)]
      if not relevant:
        print(f'No runs for {method} on {task}')
        continue
      lo = min([min(run['xs']) for run in relevant])
      hi = max([max(run['xs']) for run in relevant])
      borders = np.linspace(lo, hi, 30)
      scores = []
      for run in relevant:
        scores.append(binning(run['xs'], run['ys'], borders, fill='last')[1])
      mean = np.nanmean(scores, 0)
      std = np.nanstd(scores, 0)
      curve(
          ax, borders[1:], mean, mean - std, mean + std,
          label=method, order=j, color=COLORS[j])

    ax.tick_params(
        axis='both', which='major', labelsize='small', pad=1, length=1)
    ax.ticklabel_format(
        axis='x', style='sci', scilimits=(-2, 2))
    legend(fig, adjust=1)

  filename = outdir / (suite + '.png')
  fig.savefig(filename, dpi=300)
  print('Saved', filename)
  print('')
