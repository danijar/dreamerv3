import gzip
import json
import pathlib
import re
import sys

filenames = list((pathlib.Path(__file__).parent / 'data').glob('*.json.gz'))

if len(sys.argv) > 1:
  filenames = [x for x in filenames if re.search(sys.argv[1], x.name)]

for filename in sorted(filenames):
  print('-' * 79)
  print(filename.name)
  print('-' * 79)
  with gzip.open(filename) as f:
    runs = json.load(f)
  tasks = sorted(set(run['task'] for run in runs))
  methods = sorted(set(run['method'] for run in runs))
  seeds = sorted(set(run['seed'] for run in runs))
  print(f'Methods ({len(methods)}):', ', '.join(methods))
  print(f'Seeds ({len(seeds)}):', ', '.join(seeds))
  print(f'Tasks ({len(tasks)}):', ', '.join(tasks))
  print('Possible combinations:', len(tasks) * len(methods) * len(seeds))
  print('Runs:', len(runs))
  print('')
