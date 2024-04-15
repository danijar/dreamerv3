import re
import sys

from . import config


class Flags:

  def __init__(self, *args, **kwargs):
    self._config = config.Config(*args, **kwargs)

  def parse(self, argv=None, help_exits=True):
    parsed, remaining = self.parse_known(argv)
    for flag in remaining:
      if flag.startswith('--') and flag[2:] not in self._config.flat:
        raise KeyError(f"Flag '{flag}' did not match any config keys.")
    if remaining:
      raise ValueError(
          f'Could not parse all arguments. Remaining: {remaining}')
    return parsed

  def parse_known(self, argv=None, help_exits=False):
    if argv is None:
      argv = sys.argv[1:]
    if '--help' in argv:
      print('\nHelp:')
      lines = str(self._config).split('\n')[2:]
      print('\n'.join('--' + re.sub(r'[:,\[\]]', '', x) for x in lines))
      help_exits and sys.exit()
    parsed = {}
    remaining = []
    key = None
    vals = None
    for arg in argv:
      if arg.startswith('--'):
        if key:
          self._submit_entry(key, vals, parsed, remaining)
        if '=' in arg:
          key, val = arg.split('=', 1)
          vals = [val]
        else:
          key, vals = arg, []
      else:
        if key:
          vals.append(arg)
        else:
          remaining.append(arg)
    self._submit_entry(key, vals, parsed, remaining)
    parsed = self._config.update(parsed)
    return parsed, remaining

  def _submit_entry(self, key, vals, parsed, remaining):
    if not key and not vals:
      return
    if not key:
      vals = ', '.join(f"'{x}'" for x in vals)
      remaining.extend(vals)
      return
      # raise ValueError(f"Values {vals} were not preceded by any flag.")
    name = key[len('--'):]
    if '=' in name:
      remaining.extend([key] + vals)
      return
    if not vals:
      remaining.extend([key])
      return
      # raise ValueError(f"Flag '{key}' was not followed by any values.")
    if name.endswith('+') and name[:-1] in self._config:
      key = name[:-1]
      default = self._config[key]
      if not isinstance(default, tuple):
        raise TypeError(
            f"Cannot append to key '{key}' which is of type "
            f"'{type(default).__name__}' instead of tuple.")
      if key not in parsed:
        parsed[key] = default
      parsed[key] += self._parse_flag_value(default, vals, key)
    elif self._config.IS_PATTERN.fullmatch(name):
      pattern = re.compile(name)
      keys = [k for k in self._config.flat if pattern.fullmatch(k)]
      if keys:
        for key in keys:
          parsed[key] = self._parse_flag_value(self._config[key], vals, key)
      else:
        remaining.extend([key] + vals)
    elif name in self._config:
      key = name
      parsed[key] = self._parse_flag_value(self._config[key], vals, key)
    else:
      remaining.extend([key] + vals)

  def _parse_flag_value(self, default, value, key):
    value = value if isinstance(value, (tuple, list)) else (value,)
    if isinstance(default, (tuple, list)):
      if len(value) == 1 and ',' in value[0]:
        value = value[0].split(',')
      return tuple(self._parse_flag_value(default[0], [x], key) for x in value)
    if len(value) != 1:
      raise TypeError(
          f"Expected a single value for key '{key}' but got: {value}")
    value = str(value[0])
    if default is None:
      return value
    if isinstance(default, bool):
      try:
        return bool(['False', 'True'].index(value))
      except ValueError:
        message = f"Expected bool but got '{value}' for key '{key}'."
        raise TypeError(message)
    if isinstance(default, int):
      try:
        value = float(value)  # Allow scientific notation for integers.
        assert float(int(value)) == value
      except (ValueError, TypeError, AssertionError):
        message = f"Expected int but got '{value}' for key '{key}'."
        raise TypeError(message)
      return int(value)
    if isinstance(default, dict):
      raise KeyError(
          f"Key '{key}' refers to a whole dict. Please speicfy a subkey.")
    try:
      return type(default)(value)
    except ValueError:
      raise TypeError(
          f"Cannot convert '{value}' to type '{type(default).__name__}' for "
          f"key '{key}'.")
