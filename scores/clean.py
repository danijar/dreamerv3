import gzip
import json
import pathlib
import re
import sys

filenames = list((pathlib.Path(__file__).parent / 'data').glob('*.json.gz'))

if len(sys.argv) > 1:
  filenames = [x for x in filenames if re.search(sys.argv[1], x.name)]

for filename in sorted(filenames):
  print(filename.name)

  with gzip.open(filename, 'rb') as f:
    runs = original = json.load(f)
  edited = False

  runs = [r for r in runs if not r['task'].startswith('stats_')]

  tasks = sorted(set(run['task'] for run in runs))
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))

  new = sorted([str(x) for x in range(len(seeds))])
  renames = {k: v for k, v in zip(seeds, new) if k != v}
  for run in runs:
    if run['seed'] in renames:
      run['seed'] = renames[run['seed']]
      edited = True

  if filename.name.startswith('atari200m'):
    for run in runs:
      if run['task'] == 'atari_james_bond':
        run['task'] = 'atari_jamesbond'
        edited = True

  # if filename.name.startswith('...'):
  #   for run in runs:
  #     keep = len([x for x in run['xs'] if x <= 1e6])
  #     if keep < len(run['xs']):
  #       run['xs'] = run['xs'][:keep]
  #       run['ys'] = run['ys'][:keep]
  #       edited = True

  runs = sorted(runs, key=lambda x: ((x['task'], x['method'], x['seed'])))

  if (runs != original) or edited:
    print(f'Writing changes')
    with gzip.open(filename, 'wb') as f:
      f.write(json.dumps(runs).encode('utf-8'))
