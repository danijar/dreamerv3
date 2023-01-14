import builtins

import numpy as np

try:
  import rich.console
  console = rich.console.Console()
except ImportError:
  console = None


CONVERSION = {
    np.floating: np.float32,
    np.signedinteger: np.int64,
    np.uint8: np.uint8,
    bool: bool,
}


def convert(value):
  value = np.asarray(value)
  if value.dtype not in CONVERSION.values():
    for src, dst in CONVERSION.items():
      if np.issubdtype(value.dtype, src):
        if value.dtype != dst:
          value = value.astype(dst)
        break
    else:
      raise TypeError(f"Object '{value}' has unsupported dtype: {value.dtype}")
  return value


def print(value):
  global console
  if console:
    console.print(value)
  else:
    builtins.print(value)
