from datetime import datetime


def timestamp(now=None, millis=False):
  now = datetime.now() if now is None else now
  string = now.strftime("%Y%m%dT%H%M%S")
  if millis:
    string += f'F{now.microsecond:06d}'
  return string
