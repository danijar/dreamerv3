import re

try:
  import colored
except ImportError:
  print('For colored outputs: pip install colored')
  colored = None


REGEX_TOKEN = re.compile(
    r"([^a-zA-Z0-9-_./'\"\[\]]|['\"][^\s]*['\"])", re.MULTILINE)
REGEX_NUMBER = re.compile(
    r'([-+]?[0-9]+[0-9.,]*(e[-+]?[0-9])?|nan|-?inf)')
KEYWORDS = (
    'True', 'False', 'None', 'bool', 'int', 'str', 'float',
    'uint8', 'float16', 'float32', 'int32', 'int64')


def print_(*values, color=True, **kwargs):
  value = kwargs.get('sep', ' ').join(str(x) for x in values)
  assert not color or isinstance(color, (bool, str)), color
  if isinstance(color, str) and colored:
    value = colored.stylize(value, colored.fg(color))
  elif color is True and colored:
    result = []
    prev = [None, None, None]  # Color, highlighted, bold
    tokens = REGEX_TOKEN.split(value) + [None]
    for i, token in enumerate(tokens[:-1]):
      new = prev.copy()
      word = token.strip()
      new[2] = None
      if not word:
        new[0] = None
      elif word in '/-+':
        new[0] = 'green'
        new[2] = True
      elif word in '{}()<>,:':
        new[0] = 'white'
      elif token == '=':
        new[0] = 'white'
      elif word[0].isalpha() and tokens[i + 1] == '=':
        new[0] = 'magenta'
      elif word in KEYWORDS:
        new[0] = 'blue'
      elif word.startswith('---'):
        new[1] = True
      elif REGEX_NUMBER.match(word):
        new[0] = 'blue'
      elif word[0] == word[-1] == "'":
        new[0] = 'red'
      elif word[0] == word[-1] == '"':
        new[0] = 'red'
      elif word[0] == '[' and word[-1] == ']':
        new[0] = 'cyan'
      elif any(word.startswith(x) for x in ('/', '~', './')):
        new[0] = 'yellow'
      elif len(word) >= 3 and word[0] == word[-1] and word[0] in ("'", '"'):
        new[0] = 'green'
      elif word[0] == word[0].upper():
        new[0] = None
      else:
        new[0] = None
      if new[1]:
        new[0] = 'cyan'
        new[2] = True
      if new != prev:
        result.append(colored.attr('reset'))
        new[0] and result.append(colored.fg(new[0]))
        new[2] and result.append(colored.attr('bold'))
      result.append(token)
      prev = new
      if '\n' in token:
        prev[1] = None
        prev[2] = None
    result.append(colored.attr('reset'))
    value = ''.join(result)
  print(value, **kwargs)


def format_(value):
  if isinstance(value, dict):
    items = [f'{format_(k)}: {format_(v)}' for k, v in value.items()]
    return '{' + ', '.join(items) + '}'
  if isinstance(value, list):
    return '[' + ', '.join(f'{format_(x)}' for x in value) + ']'
  if isinstance(value, tuple):
    return '(' + ', '.join(f'{format_(x)}' for x in value) + ')'
  if hasattr(value, 'shape') and hasattr(value, 'dtype'):
    shape = ','.join(str(x) for x in value.shape)
    dtype = value.dtype.name
    for long, short in {'float': 'f', 'uint': 'u', 'int': 'i'}.items():
      dtype = dtype.replace(long, short)
    return f'{dtype}<{shape}>'
  if isinstance(value, bytes):
    value = '0x' + value.hex() if r'\x' in str(value) else str(value)
    if len(value) > 32:
      value = value[:32 - 3] + '...'
  return str(value)
