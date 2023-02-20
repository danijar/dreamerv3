import contextlib
import glob
import os
import re
import shutil


class Path:

  filesystems = []

  def __new__(cls, path):
    path = str(path)
    for impl, pred in cls.filesystems:
      if pred(path):
        obj = super().__new__(impl)
        obj.__init__(path)
        return obj
    raise NotImplementedError(f'No filesystem supports: {path}')

  def __getnewargs__(self):
    return (self._path,)

  def __init__(self, path):
    assert isinstance(path, str)
    path = re.sub(r'^\./*', '', path)  # Remove leading dot or dot slashes.
    path = re.sub(r'(?<=[^/])/$', '', path)  # Remove single trailing slash.
    path = path or '.'  # Empty path is represented by a dot.
    self._path = path

  def __truediv__(self, part):
    sep = '' if self._path.endswith('/') else '/'
    return type(self)(f'{self._path}{sep}{str(part)}')

  def __repr__(self):
    return f'Path({str(self)})'

  def __fspath__(self):
    return str(self)

  def __eq__(self, other):
    return self._path == other._path

  def __lt__(self, other):
    return self._path < other._path

  def __str__(self):
    return self._path

  @property
  def parent(self):
    if '/' not in self._path:
      return type(self)('.')
    parent = self._path.rsplit('/', 1)[0]
    parent = parent or ('/' if self._path.startswith('/') else '.')
    return type(self)(parent)

  @property
  def name(self):
    if '/' not in self._path:
      return self._path
    return self._path.rsplit('/', 1)[1]

  @property
  def stem(self):
    return self.name.split('.', 1)[0] if '.' in self.name else self.name

  @property
  def suffix(self):
    return ('.' + self.name.split('.', 1)[1]) if '.' in self.name else ''

  def read(self, mode='r'):
    assert mode in 'r rb'.split(), mode
    with self.open(mode) as f:
      return f.read()

  def write(self, content, mode='w'):
    assert mode in 'w a wb ab'.split(), mode
    with self.open(mode) as f:
      f.write(content)

  @contextlib.contextmanager
  def open(self, mode='r'):
    raise NotImplementedError

  def absolute(self):
    raise NotImplementedError

  def glob(self, pattern):
    raise NotImplementedError

  def exists(self):
    raise NotImplementedError

  def isfile(self):
    raise NotImplementedError

  def isdir(self):
    raise NotImplementedError

  def mkdirs(self):
    raise NotImplementedError

  def remove(self):
    raise NotImplementedError

  def rmtree(self):
    raise NotImplementedError

  def copy(self, dest):
    raise NotImplementedError

  def move(self, dest):
    self.copy(dest)
    self.remove()


class LocalPath(Path):

  def __init__(self, path):
    super().__init__(os.path.expanduser(str(path)))

  @contextlib.contextmanager
  def open(self, mode='r'):
    with open(str(self), mode=mode) as f:
      yield f

  def absolute(self):
    return type(self)(os.path.absolute(str(self)))

  def glob(self, pattern):
    for path in glob.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return os.path.exists(str(self))

  def isfile(self):
    return os.path.isfile(str(self))

  def isdir(self):
    return os.path.isdir(str(self))

  def mkdirs(self):
    os.makedirs(str(self), exist_ok=True)

  def remove(self):
    os.rmdir(str(self)) if self.isdir() else os.remove(str(self))

  def rmtree(self):
    shutil.rmtree(self)

  def copy(self, dest):
    if self.isfile():
      shutil.copy(self, type(self)(dest))
    else:
      shutil.copytree(self, type(self)(dest), dirs_exist_ok=True)

  def move(self, dest):
    shutil.move(self, dest)


class GFilePath(Path):

  def __init__(self, path):
    path = str(path)
    if not (path.startswith('/') or '://' in path):
      path = os.path.abspath(os.path.expanduser(path))
    super().__init__(path)
    import tensorflow as tf
    self._gfile = tf.io.gfile

  @contextlib.contextmanager
  def open(self, mode='r'):
    path = str(self)
    if 'a' in mode and path.startswith('/cns/'):
      path += '%r=3.2'
    if mode.startswith('x') and self.exists():
      raise FileExistsError(path)
      mode = mode.replace('x', 'w')
    with self._gfile.GFile(path, mode) as f:
      yield f

  def absolute(self):
    return self

  def glob(self, pattern):
    for path in self._gfile.glob(f'{str(self)}/{pattern}'):
      yield type(self)(path)

  def exists(self):
    return self._gfile.exists(str(self))

  def isfile(self):
    return self.exists() and not self.isdir()

  def isdir(self):
    return self._gfile.isdir(str(self))

  def mkdirs(self):
    self._gfile.makedirs(str(self))

  def remove(self):
    self._gfile.remove(str(self))

  def rmtree(self):
    self._gfile.rmtree(str(self))

  def copy(self, dest):
    self._gfile.copy(str(self), str(dest), overwrite=True)

  def move(self, dest):
    dest = Path(dest)
    if dest.isdir():
      dest.rmtree()
    self._gfile.rename(self, str(dest), overwrite=True)


Path.filesystems = [
    (GFilePath, lambda path: path.startswith('gs://')),
    (GFilePath, lambda path: path.startswith('/cns/')),
    (LocalPath, lambda path: True),
]
