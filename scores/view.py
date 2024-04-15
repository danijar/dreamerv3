import pathlib
import sys

import pandas as pd


scoredir = pathlib.Path(__file__).parent

total_size = 0

prefix = None

for filename in sorted(list(scoredir.glob('*.json.gz'))):

  new_prefix = filename.name.split('-')[0]
  if prefix != new_prefix:
    print(f'\n{new_prefix}')
    prefix = new_prefix

  df = pd.read_json(filename)

  budget = df['xs'].apply(max).max()
  man, exp = f'{budget:.1e}'.split('e')
  budget = f'{man}e{str(int(exp)):<2}'

  points = df['xs'].apply(len).mean()
  size = filename.stat().st_size

  print(f'  {filename.name:<45}', '   '.join([
      f'{len(df):3d} runs',
      f'{len(df.task.unique()):2d} tasks',
      f'{len(df.seed.unique()):2d} seeds',
      f'{budget} budget',
      f'{points:5.0f} bins',
      f'{size / 1024:5.0f} kB',
  ]))

  total_size += size

print(f'\nTotal size: {total_size / (1024 ** 2):.0f} Mb')
