import contextlib
import functools
import inspect
import re
import threading

import jax
import jax.numpy as jnp

__version__ = '2.3.1'


###############################################################################
# State
###############################################################################


# When running an impure function that accesses state, it will find the state
# in this global variable. The pure() wrapper populates this global variable
# with the provided state, calls the inner function, and then the takes the
# resulting state out of the global variable to return it back to the user.
# To allow multi-threaded programs to use impure functions in parallel, the
# context is a dictionary with a slot for each thread identifier.
CONTEXT = {}


class Context(dict):

  def __init__(
      self, entries, seed, create, modify, ignore, reserve, name):
    super().__init__(entries)
    self.create = create   # Allow creating new state entries.
    self.modify = modify   # Allow modifying existing state entries.
    self.ignore = ignore   # Ignore modifications to existing state entries.
    self.seed = seed
    self.reserve = reserve
    self.name = name
    self.accessed = set()  # Keys accessed for reading.
    self.created = set()   # Keys accessed for creating.
    self.modified = set()  # Keys accessed for modifying (even if ignored).

  def update(self, entries):
    for key, value in dict(entries).items():
      self[key] = value

  def __getitem__(self, key):
    self.accessed.add(key)
    try:
      return super().__getitem__(key)
    except KeyError:
      raise KeyError(
          f"Trying to access state key '{key}' that does not exist in context "
          f'create={self.create} modify={self.modify} ignore={self.ignore}.')

  def __setitem__(self, key, value):
    if key in self:
      self.modified.add(key)
    else:
      self.created.add(key)
    if self.ignore and key in self:
      return  # Do not overwrite existing entries.
    if not self.create and key not in self:
      raise RuntimeError(
          'Pass create=True to pure functions to allow them to create new '
          f'state entries or use nj.init(). You were trying to set {key} to '
          f'shape {value.shape}.')
    if not self.modify:
      existing = self[key]
      raise RuntimeError(
          'Cannot modify state entries here. (If you want to modify '
          'state inside of scan() pass modify=True.) ' +
          f'You were trying to change {key} from shape {existing.shape} '
          f'and dtype {existing.dtype} to shape {value.shape} and ' +
          f'dtype {value.dtype}.')
    super().__setitem__(key, value)


def pure(fun, nested=False):
  """Wrap an impure function that uses global state to explicitly pass the
  state in and out. The result is a pure function that is composable with JAX
  transformation. The pure function can be used as follows:
  ```
  state, out = fun(state, *args, **kwargs)
  ```
  Additional keyword arguments can be provided:
  - `seed`: Provide an integer or array of two integers to be able to use
    `nj.seed()` inside the impure function.
  - `create=False`: Boolean indicating whether the impure function will be
    allowed to create new state entries.
  - `modify=True`: Boolean indicating whether the impure function will be
    allowed to modify existing state entries.
  - `ignore=False`: Boolean indicating whether state modifications by the
    impure function will be ignored silently; useful for initialization.
  - `track=False`: Boolean indicating whether to return the sets of state
    keys that the impure function attempted to read, modify, and create.
  """
  def purified(
      state, *args, seed=None, create=None, modify=None, ignore=None,
      track=False, **kwargs):
    if isinstance(seed, int) or (hasattr(seed, 'shape') and seed.shape == ()):
      seed = jnp.array([seed, seed], jnp.uint32)
    context = CONTEXT.get(threading.get_ident(), None)
    if context is not None:
      create = create if create is not None else context.create
      modify = modify if modify is not None else context.modify
      ignore = ignore if ignore is not None else context.ignore
      assert context.create or not create, 'Parent context disabled create.'
      assert context.modify or not modify, 'Parent context disabled modify.'
      assert not context.ignore or ignore, 'Parent context enabled ignore.'
    else:
      create = create if create is not None else False
      modify = modify if modify is not None else True
      ignore = ignore if ignore is not None else False
    if not isinstance(state, dict):
      raise ValueError('Must provide a dict as state.')
    name = getattr(fun, '__name__', str(fun))
    if context and (not nested):
      raise RuntimeError(
          f'You are trying to call pure {name}() inside pure '
          f'{context.name}(). Is that intentional? If you want to nest pure '
          f'functions, use pure(..., nested=True) for the inner function.')
    before = context
    try:
      context = Context(
          state.copy(), seed, create, modify, ignore, [], name)
      CONTEXT[threading.get_ident()] = context
      out = fun(*args, **kwargs)
      state = dict(context)
      if before:
        before.accessed |= context.accessed
        before.modified |= context.modified
        before.created |= context.created
      if track:
        return state, out, context.accessed, context.modified, context.created
      return state, out
    finally:
      CONTEXT[threading.get_ident()] = before
  purified._is_pure = True
  return purified


def context():
  """Access and modify the global context from within an impure function. For
  advanced users only. Prefer to use module methods to access and modify state
  and seed() to get the next random seed."""
  context = CONTEXT.get(threading.get_ident(), None)
  if context is None:
    raise RuntimeError('Wrap impure functions in pure() before running them.')
  return context


def init(fun, **jit_kwargs):
  """Creates an initializer for a pure or impure function, which when called
  with example inputs , quickly populates the initial state without performing
  the actual computation of the function."""
  if not getattr(fun, '_is_pure', False):
    fun = pure(fun)
  def wrapper(*args, **kwargs):
    state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
    del out
    return state
  return jax.jit(wrapper, **jit_kwargs)


@jax.named_scope('seed')
def seed(amount=None, optional=False, reserve=16):
  """Split the global random seed and return a new local seed."""
  ctx = context()
  if ctx.seed is None:
    if optional:
      return None if amount is None else [None] * amount
    raise ValueError(
        'You must provide a seed to the pure function to use nj.seed() '
        'inside the impure function.')
  if amount:
    keys = jax.random.split(ctx.seed, amount + 1)
    ctx.seed = keys[0]
    return keys[1:]
  else:
    if not ctx.reserve:
      keys = jax.random.split(ctx.seed, reserve)
      ctx.seed = keys[0]
      ctx.reserve = list(keys[1:])
    return ctx.reserve.pop(0)


def creating():
  """Indicates whether the program is currently allowed to create state
  entries. Can use used for initialization logic that should be excluded from
  compiled functions."""
  return context().create


###############################################################################
# Transformations
###############################################################################


@jax.named_scope('grad')
def grad(fun, keys, has_aux=False):
  """Compute the gradient of an impure function with respect to the specified
  state entries or modules. The transformed function returns a tuple containing
  the computed value, selected state entries, their gradients, and if
  applicable auxiliary outputs of the function."""
  keys = keys if hasattr(keys, '__len__') else (keys,)
  if not has_aux:
    fun = lambda *args, _fun=fun, **kwargs: (_fun(*args, *kwargs), {})
  fun = pure(fun, nested=True)

  def wrapper(*args, **kwargs):
    accessed, modified = _prerun(fun, *args, **kwargs)

    strs = []
    for key in keys:
      if isinstance(key, Module):
        matches = key.find()
      if isinstance(key, str):
        pattern = re.compile(f'^{key}(/.*|$)')
        matches = [k for k in context() if pattern.match(k)]
      if not matches:
        raise KeyError(
            f"Gradient key '{key}' did not match any state entries. "
            'List existing entries using print(nj.context().keys()).')
      strs += matches
    existing = context().keys()
    assert all(key in existing for key in strs), (strs, existing)
    x1 = {k: v for k, v in context().items() if k in strs}
    x2 = {k: v for k, v in context().items() if k not in strs}
    assert x1

    for key in x1.keys():
      if key not in accessed:
        raise RuntimeError(
            f"Trying to compute gradient with respect to key '{key}' "
            'but the differentiated function does not access it.\n'
            f'Accessed keys: {list(accessed)}\n'
            f'Gradient keys: {list(strs)}')
    x1 = {k: v for k, v in x1.items() if k in accessed}
    x2 = {k: v for k, v in x2.items() if k in accessed}

    def forward(x1, x2, *args, **kwargs):
      before = {**x1, **x2}
      state, (y, aux) = fun(before, *args, create=False, **kwargs)
      changes = {k: v for k, v in state.items() if k in modified}
      return y, (changes, aux)
    backward = jax.value_and_grad(forward, has_aux=True)

    (y, (changes, aux)), dx = backward(
        x1, x2, *args, seed=seed(None, True), **kwargs)
    if context().modify:
      context().update(changes)
    return (y, x1, dx, aux) if has_aux else (y, x1, dx)
  return wrapper


@jax.named_scope('cond')
def cond(pred, true_fun, false_fun, *operands):
  true_fun = pure(true_fun, nested=True)
  false_fun = pure(false_fun, nested=True)

  accessed1, modified1 = _prerun(true_fun, *operands)
  accessed2, modified2 = _prerun(false_fun, *operands)
  accessed = accessed1 | accessed2
  modified = modified1 | modified2

  def true_fun_wrapper(state, seed1, seed2, *args):
    state, outs = true_fun(state, *args, seed=seed1)
    changes = {k: v for k, v in state.items() if k in modified}
    return changes, outs

  def false_fun_wrapper(state, seed1, seed2, *args):
    state, outs = false_fun(state, *args, seed=seed2)
    changes = {k: v for k, v in state.items() if k in modified}
    return changes, outs

  needed = {k: v for k, v in context().items() if k in accessed}
  changes, out = jax.lax.cond(
      pred, true_fun_wrapper, false_fun_wrapper,
      needed, *seed(2, True), *operands)
  if context().modify:
    context().update(changes)
  return out


@jax.named_scope('scan')
def scan(fun, carry, xs, reverse=False, unroll=1, axis=0):
  if axis:
    xs = jax.tree_util.tree_map(lambda x: x.swapaxes(0, axis), xs)

  fun = pure(fun, nested=True)
  accessed, modified = _prerun(
      fun, carry, jax.tree_util.tree_map(lambda x: x[0], xs))

  changing = {k: v for k, v in context().items() if k in modified}
  unchanging = {
      k: v for k, v in context().items()
      if k in accessed and k not in modified}

  def inner(carry, x):
    carry, changing = carry
    x, seed = x
    state = {**unchanging, **changing}
    state, (carry, y) = fun(state, carry, x, create=False, seed=seed)
    changing = {k: v for k, v in state.items() if k in modified}
    return (carry, changing), y

  length = len(jax.tree_util.tree_leaves(xs)[0])
  seeds = seed(length, True)
  (carry, changes), ys = jax.lax.scan(
      inner, (carry, changing), (xs, seeds), length, reverse, unroll)

  if context().modify:
    context().update(changes)

  if axis:
    ys = jax.tree_util.tree_map(lambda y: y.swapaxes(0, axis), ys)
  return carry, ys


def checkpoint(fun, **cp_kwargs):
  static = cp_kwargs.get('static_argnums', tuple())
  static = static if isinstance(static, tuple) else (static,)
  static = tuple(x + 1 for x in static)
  cp_kwargs['static_argnums'] = static

  accessed, modified = [None], [None]
  fun = pure(fun, nested=True)

  @functools.partial(jax.checkpoint, **cp_kwargs)
  def inner(*args, **kwargs):
    state, output = fun(*args, **kwargs)
    changes = {k: v for k, v in state.items() if k in modified[0]}
    return changes, output

  @jax.named_scope('checkpoint')
  def outer(*args, **kwargs):
    accessed[0], modified[0] = _prerun(fun, *args, **kwargs)
    needed = {k: v for k, v in context().items() if k in accessed[0]}
    changes, output = inner(needed, *args, seed=seed(None, True), **kwargs)
    if context().modify:
      context().update(changes)
    return output

  return outer


@jax.named_scope('prerun')
def _prerun(fun, *args, **kwargs):
  if not context().modify and not context().create:
    return set()
  state, output, accessed, modified, created = fun(
      dict(context()), *args, ignore=True, track=True,
      seed=seed(None, True), **kwargs)
  del output
  creations = {k: v for k, v in state.items() if k in created}
  context().update(creations)
  return accessed, modified


###############################################################################
# Modules
###############################################################################


SCOPE = ''


@contextlib.contextmanager
def scope(name, absolute=False):
  """Enter a relative or absolute name scope. Name scopes are used to make
  names of state entries unique."""
  global SCOPE
  if SCOPE is None:
    raise RuntimeError(
        'Purify stateful functions with fn = pure(fn) before running them.')
  outside = SCOPE
  if absolute:
    SCOPE = name
  elif SCOPE == '':
    SCOPE = name
  else:
    SCOPE = outside + '/' + name
  try:
    yield SCOPE
  except Exception as e:
    if not hasattr(e, '_njscope'):
      e._njscope = SCOPE
      if hasattr(e, 'add_note'):
        e.add_note(f"This happened inside Ninjax scope '{SCOPE}'.")
      else:
        print(f"Exception happened inside Ninjax scope '{SCOPE}'.")
    raise
  finally:
    SCOPE = outside


class ModuleMeta(type):
  """Meta class that creates a unique path for each module instance and wraps
  the methods and properties of the module to enter the name scope."""

  def __new__(mcs, name, bases, clsdict):
    """This runs once per user module class definition. It wraps the methods of
    the module class to automatically enter the name scope of the module."""
    method_names = []
    for key, value in clsdict.items():
      if key.startswith('__') and key != '__call__':
        continue
      elif isinstance(value, property):
        clsdict[key] = property(
            value.fget if not value.fget else _scope_method(value.fget),
            value.fset if not value.fset else _scope_method(value.fset),
            value.fdel if not value.fdel else _scope_method(value.fdel),
            doc=value.__doc__)
      elif inspect.isfunction(value):
        method_names.append(key)
    cls = super(ModuleMeta, mcs).__new__(mcs, name, bases, clsdict)
    cls.__field_defaults = {
        k: getattr(cls, k) for k, v in cls.__annotations__.items()
        if hasattr(cls, k)}
    for key, value in cls.__annotations__.items():
      setattr(cls, key, property(lambda self, key=key: self.__fields[key]))
    for method_name in method_names:
      method = getattr(cls, method_name)
      method = _scope_method(method)
      setattr(cls, method_name, method)
    return cls

  def __call__(cls, *args, name=None, **kwargs):
    """This runs once per use module instance creation. It derives a unique
    name and path for the module instance."""
    if not isinstance(name, str):
      raise TypeError(
          "Please provide a module name via Module(..., name='example').")
    if not re.match(r'^[A-Za-z0-9_]+$', name):
      raise ValueError(
          'Only letters, numbers, and underscores are allowed in scope names; '
          f'got: {name}')
    fields = {}
    for key, typ in cls.__annotations__.items():
      if key in kwargs:
        value = kwargs.pop(key)
      elif key in cls.__field_defaults:
        value = cls.__field_defaults[key]
      else:
        raise TypeError(
            f"Pass a keyword argument for field '{key}' or define a default.")
      if typ is not None and not isinstance(value, typ):
        raise TypeError(
            f"Value '{value}' for field '{key}' is not of type "
            f"'{typ.__name__}'.")
      fields[key] = value
    obj = cls.__new__(cls)
    obj.__fields = fields
    with scope(name) as path:
      obj._path = path
    obj._submodules = {}
    init = _scope_method(cls.__init__)
    init(obj, *args, **kwargs)
    return obj


def _scope_method(method):
  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    with scope(self._path, absolute=True):
      with jax.named_scope(self._path.split('/')[-1]):
        return method(self, *args, **kwargs)
  return wrapper


class Module(object, metaclass=ModuleMeta):
  """Base class for users to inherit their modules from. Provides automatic
  name scoping via the meta class and helper functions for accessing state."""

  def __repr__(self):
    return f'{self.__class__.__name__}({self.path})'

  @property
  def path(self):
    """The unique name scope of this module instance as a string."""
    return self._path

  @property
  def name(self):
    """The name of this module instance as a string."""
    return self._path.split('/')[-1]

  def get(self, name, *args, **kwargs):
    """Retrieve or create a state entry that belongs to this module."""
    assert '{' not in name, 'Did you forget to format a string?'
    path = self.path + '/' + name
    if name in self._submodules:
      return self._submodules[name]
    if path in context():
      return context()[path]
    ctor, *args = args
    if 'name' in inspect.signature(ctor).parameters:
      kwargs['name'] = name
    value = ctor(*args, **kwargs)
    # We support trees of arrays for easier integration with other libraries.
    flat = jax.tree_util.tree_leaves(value)
    if all(isinstance(x, jnp.ndarray) for x in flat):
      context()[path] = value
      # Look up the value again to make sure we are registering it as an
      # accessed key in the context.
      return context()[path]
    else:
      self._submodules[name] = value
      return value

  def put(self, *args):
    """Update or create state entries that belong to this module. The arguments
    are either string name and value of a single state entry or a dict holding
    multiple state entries."""
    if len(args) == 2:
      name, value = args
      self.put({self.path + '/' + name: value})
      return value
    assert len(args) == 1 and isinstance(args[0], dict)
    mapping = args[0]
    prefix = self.path + '/'
    for key in mapping:
      if not key.startswith(prefix):
        raise KeyError(f'Key {key} does not belong to module {self.path}.')
    context().update(mapping)

  def find(self, pattern=r'.*', empty_ok=False):
    """Find the state entries of this module, optionally filtered by regex."""
    pattern = re.compile(pattern)
    prefix = self.path + '/'
    results = {}
    for key, value in context().items():
      if not key.startswith(prefix):
        continue
      if pattern.match(key[len(prefix):]):
        results[key] = value
    if not empty_ok and not results:
      raise KeyError(f'Pattern {pattern} matched no state keys.')
    return results


class Variable(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.ctor = ctor
    self.args = args
    self.kwargs = kwargs

  def read(self):
    return self.get('value', self.ctor, *self.args, **self.kwargs)

  def write(self, value):
    return self.put('value', value)


###############################################################################
# Integrations
###############################################################################


def FromFlax(ctor):
  class FlaxModule(Module):
    def __init__(self, *args, **kwargs):
      self.module = ctor(*args, **kwargs)
    def __call__(self, *args, **kwargs):
      seed_ = seed() if creating() else None
      state = self.get('flax', self.module.init, seed_, *args, **kwargs)
      return self.module.apply(state, *args, **kwargs)
  return FlaxModule


def FromHaiku(ctor):
  class HaikuModule(Module):
    def __init__(self, *args, **kwargs):
      import haiku as hk
      def net(*args_, **kwargs_):
        return ctor(*args, **kwargs)(*args_, **kwargs_)
      self.transformed = hk.transform(net)
    def __call__(self, *args, **kwargs):
      seed_ = seed() if creating() else None
      state = self.get('haiku', self.transformed.init, seed_, *args, **kwargs)
      return self.transformed.apply(state, seed_, *args, **kwargs)
  return HaikuModule


def FromOptax(ctor):
  class OptaxModule(Module):
    def __init__(self, *args, **kwargs):
      self.opt = ctor(*args, **kwargs)
    def __call__(self, loss, keys, *args, **kwargs):
      import optax
      loss, params, grads = grad(loss, keys)(*args, **kwargs)
      optstate = self.get('state', self.opt.init, params)
      updates, optstate = self.opt.update(grads, optstate)
      self.put('state', optstate)
      context().update(optax.apply_updates(params, updates))
      return loss, params, grads
  return OptaxModule
