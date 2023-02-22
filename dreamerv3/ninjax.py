import contextlib
import functools
import inspect
import re
import threading
from functools import partial as bind

import jax
import jax.numpy as jnp

__version__ = '0.9.0'


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

  def __init__(self, entries, rng, create, modify, ignore, reserve, name):
    super().__init__(entries)
    self.create = create  # Allow creating new state entries.
    self.modify = modify  # Allow modifying existing state entries.
    self.ignore = ignore  # Ignore modifications to existing state entries.
    self.rng = rng
    self.reserve = reserve
    self.name = name

  def update(self, entries):
    for key, value in dict(entries).items():
      self[key] = value

  def __setitem__(self, key, value):
    if not self.modify:
      raise RuntimeError(
          'Cannot modify state entries here. If you want to modify '
          'state inside of scan() set modify=True. ' +
          f'You were trying to set {key} to shape {value.shape} and ' +
          f'dtype {value.dtype}.')
    if self.ignore and key in self:
      return  # Do not overwrite existing entries.
    if not self.create and key not in self:
      raise RuntimeError(
          'Can only create state entries during first call. ' +
          f'You were trying to set {key} to shape {value.shape} and ' +
          f'dtype {value.dtype}.')
    super().__setitem__(key, value)


def pure(fun, nested=False):
  """Wrap an impure function that uses global state to explicitly pass the
  state in and out. The result is a pure function that is composable with JAX
  transformation. The pure function can be used as follows:
  `out, state = fun(state, rng, *args, **kwargs)`."""
  def purified(
      state, rng, *args, create=None, modify=None, ignore=None, **kwargs):
    context = CONTEXT.get(threading.get_ident(), None)
    if context:
      create = create if create is not None else context.create
      modify = modify if modify is not None else context.modify
      ignore = ignore if ignore is not None else context.ignore
      assert context.create or not create, 'Parent context disabled create.'
      assert context.modify or not modify, 'Parent context disabled modify.'
      assert not context.ignore or ignore, 'Parent context enabled ignore.'
    else:
      create = create if create is not None else True
      modify = modify if modify is not None else True
      ignore = ignore if ignore is not None else False
    if not isinstance(state, dict):
      raise ValueError('Must provide a dict as state.')
    if context and (not nested):
      raise RuntimeError(
          f'You are trying to call pure {fun.__name__}() inside pure '
          f'{context.name}(). Is that intentional? If you want to nest pure '
          f'functions, use pure(..., nested=True) for the inner function.')
      # raise RuntimeError(
      #     f'If you want to nest run() calls, use nested=True. ({context})')
    before = context
    try:
      name = fun.__name__
      if rng.shape == ():
        rng = jax.random.PRNGKey(rng)
      context = Context(state.copy(), rng, create, modify, ignore, [], name)
      CONTEXT[threading.get_ident()] = context
      out = fun(*args, **kwargs)
      state = dict(context)
      return out, state
    finally:
      CONTEXT[threading.get_ident()] = before
  purified.pure = True
  return purified


def context():
  """Access and modify the global context from within an impure function. For
  advanced users only. Prefer to use module methods to access and modify state
  and rng() to get the next RNG key."""
  context = CONTEXT.get(threading.get_ident(), None)
  if context is None:
    raise RuntimeError('Wrap impure functions in pure() before running them.')
  return context


@jax.named_scope('rng')
def rng(amount=None, reserve=16):
  """Split the global RNG key and return a new local key."""
  ctx = context()
  if amount:
    keys = jax.random.split(ctx.rng, amount + 1)
    ctx.rng = keys[0]
    return keys[1:]
  else:
    if not ctx.reserve:
      keys = jax.random.split(ctx.rng, reserve)
      ctx.rng = keys[0]
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
  if getattr(fun, 'pure', False):
    raise ValueError('Use plain jax.grad() for pure functions.')
  if not has_aux:
    fun = lambda *args, _fun=fun, **kwargs: (_fun(*args, *kwargs), {})
  fun = pure(fun, nested=True)
  def forward(x1, x2, rng, *args, **kwargs):
    (y, aux), state = fun({**x1, **x2}, rng, *args, create=False, **kwargs)
    return y, (aux, state)
  backward = jax.value_and_grad(forward, has_aux=True)
  @functools.wraps(backward)
  def wrapper(*args, **kwargs):
    _prerun(fun, *args, **kwargs)
    assert all(isinstance(x, (str, Module)) for x in keys)
    strs = [x for x in keys if isinstance(x, str)]
    mods = [x for x in keys if isinstance(x, Module)]
    for mod in mods:
      strs += mod.getm()
    x1 = {k: v for k, v in context().items() if k in strs}
    x2 = {k: v for k, v in context().items() if k not in strs}
    (y, (aux, state)), dx = backward(x1, x2, rng(), *args, **kwargs)
    context().update(state)
    return (y, x1, dx, aux) if has_aux else (y, x1, dx)
  return wrapper


def jit(fun, static=None, **kwargs):
  """Compiles a pure function for fast execution. Only the first call of the
  function is allowed to create state entries."""
  if not getattr(fun, 'pure', False):
    raise ValueError('Use pure() before applying jit().')
  static = static or ()

  @bind(jax.jit, static_argnums=[0], **kwargs)
  def init(statics, rng, *args, **kw):
    # Return only state so JIT can remove dead code for fast initialization.
    s = fun({}, rng, *args, ignore=True, **dict(statics), **kw)[1]
    return s

  @bind(jax.jit, static_argnums=[0], **kwargs)
  def apply(statics, state, rng, *args, **kw):
    return fun(state, rng, *args, create=False, **dict(statics), **kw)

  @functools.wraps(fun)
  def wrapper(state, rng, *args, init_only=False, **kw):
    if any([name not in kw for name in static]):
      raise ValueError('Please pass all static arguments by keyword.')
    state = state.copy()
    statics = tuple(sorted([(k, v) for k, v in kw.items() if k in static]))
    kw = {k: v for k, v in kw.items() if k not in static}
    if not hasattr(wrapper, 'keys'):
      created = init(statics, rng, *args, **kw)
      wrapper.keys = set(created.keys())
      for key, value in created.items():
        if key not in state:
          state[key] = value
    if init_only:
      return state
    else:
      selected = {k: v for k, v in state.items() if k in wrapper.keys}
      out, updated = apply(statics, selected, rng, *args, **kw)
      return out, {**state, **updated}
  return wrapper


def pmap(fun, axis_name=None, static=None, **kwargs):
  """Compiles n pure function for fast execution across multiple devices. Only
  the first call of the function is allowed to create state entries."""
  if not getattr(fun, 'pure', False):
    raise ValueError('Use pure() before applying jit().')
  static = static or ()

  @bind(
      jax.pmap, axis_name=axis_name, static_broadcasted_argnums=[0], **kwargs)
  def init(statics, rng, *args, **kw):
    # Return only state so JIT can remove dead code for fast initialization.
    return fun({}, rng, *args, ignore=True, **dict(statics), **kw)[1]

  @bind(
      jax.pmap, axis_name=axis_name, static_broadcasted_argnums=[0], **kwargs)
  def apply(statics, state, rng, *args, **kw):
    return fun(state, rng, *args, create=False, **dict(statics), **kw)

  @functools.wraps(fun)
  def wrapper(state, rng, *args, init_only=False, **kw):
    if any([name not in kw for name in static]):
      raise ValueError('Please pass all static arguments by keyword.')
    state = state.copy()
    statics = tuple(sorted([(k, v) for k, v in kw.items() if k in static]))
    kw = {k: v for k, v in kw.items() if k not in static}
    if not hasattr(wrapper, 'keys'):
      created = init(statics, rng, *args, **kw)
      wrapper.keys = set(created.keys())
      for key, value in created.items():
        if key not in state:
          state[key] = value
    if init_only:
      return state
    else:
      selected = {k: v for k, v in state.items() if k in wrapper.keys}
      out, updated = apply(statics, selected, rng, *args, **kw)
      return out, {**state, **updated}
  return wrapper


@jax.named_scope('cond')
def cond(pred, true_fun, false_fun, *operands):
  true_fun = pure(true_fun, nested=True)
  false_fun = pure(false_fun, nested=True)
  _prerun(true_fun, *operands)
  _prerun(false_fun, *operands)
  out, state = jax.lax.cond(
      pred,
      lambda state, rng1, rng2, *args: true_fun(state, rng1, *args),
      lambda state, rng1, rng2, *args: false_fun(state, rng2, *args),
      dict(context()), *rng(2), *operands)
  context().update(state)
  return out


@jax.named_scope('scan')
def scan(fun, carry, xs, reverse=False, unroll=1, modify=False):
  fun = pure(fun, nested=True)
  _prerun(fun, carry, jax.tree_util.tree_map(lambda x: x[0], xs))
  length = len(jax.tree_util.tree_leaves(xs)[0])
  rngs = rng(length)
  if modify:
    def inner(carry, x):
      carry, state = carry
      x, rng = x
      (carry, y), state = fun(state, rng, carry, x, create=False)
      return (carry, state), y
    (carry, state), ys = jax.lax.scan(
        inner, (carry, dict(context())), (xs, rngs), length, reverse, unroll)
    context().update(state)
  else:
    def inner(carry, x):
      x, rng = x
      (carry, y), state = fun(
          dict(context()), rng, carry, x, create=False, modify=False)
      return carry, y
    carry, ys = jax.lax.scan(inner, carry, (xs, rngs), length, reverse, unroll)
  return carry, ys


@jax.named_scope('_prerun')
def _prerun(fun, *args, **kwargs):
  if not context().create:
    return
  discarded, state = fun(dict(context()), rng(), *args, ignore=True, **kwargs)
  # jax.tree_util.tree_map(
  #     lambda x: hasattr(x, 'delete') and x.delete(), discarded)
  context().update(state)


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
  yield SCOPE
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
    for method_name in method_names:
      method = getattr(cls, method_name)
      method = _scope_method(method)
      setattr(cls, method_name, method)
    return cls

  def __call__(cls, *args, name=None, **kwargs):
    """This runs once per use module instance creation. It derives a unique
    name and path for the module instance."""
    if not isinstance(name, str):
      raise KeyError(
          "Please provide a module name via Module(..., name='example').")
    if not re.match(r'[A-Za-z0-9_]+', name):
      raise KeyError(
          'Only letters, numbers, and underscores are allowed in scope names.')
    obj = cls.__new__(cls)
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
    path = self.path + '/' + name
    if name in self._submodules:
      return self._submodules[name]
    if path in context():
      return context()[path]
    ctor, *args = args
    if 'name' in inspect.signature(ctor).parameters:
      kwargs['name'] = name
    value = ctor(*args, **kwargs)
    flat, _ = jax.tree_util.tree_flatten(value)
    if all(isinstance(x, jnp.ndarray) for x in flat):
      context()[path] = value
    else:
      self._submodules[name] = value
    return value

  def put(self, name, value):
    """Update or create a single state entry that belongs to this module."""
    self.putm({self.path + '/' + name: value})
    return value

  def getm(self, pattern=r'.*', allow_empty=False):
    """Read the state entries of this module, optionally filtered by regex."""
    pattern = re.compile(pattern)
    prefix = self.path + '/'
    results = {}
    for key, value in context().items():
      if not key.startswith(prefix):
        continue
      if pattern.match(key[len(prefix):]):
        results[key] = value
    if not allow_empty and not results:
      raise KeyError(f'Pattern {pattern} matched no state keys.')
    return results

  def putm(self, mapping):
    """Update or create multiple state entries that belong to this module."""
    prefix = self.path + '/'
    for key in mapping:
      if not key.startswith(prefix):
        raise KeyError(f'Key {key} does not belong to module {self.path}.')
    context().update(mapping)


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


class HaikuModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    import haiku as hk
    def net(*args_, **kwargs_):
      return ctor(*args, **kwargs)(*args_, **kwargs_)
    self.transformed = hk.transform(net)

  def __call__(self, *args, **kwargs):
    state = self.get('state', self.transformed.init, rng(), *args, **kwargs)
    return self.transformed.apply(state, rng(), *args, **kwargs)


class FlaxModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.module = ctor(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    state = self.get('state', self.module.init, rng(), *args, **kwargs)
    return self.module.apply(state, *args, **kwargs)


class OptaxModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.opt = ctor(*args, **kwargs)

  def __call__(self, loss, keys, *args, **kwargs):
    import optax
    loss, params, grads = grad(loss, keys)(*args, **kwargs)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate)
    self.put('state', optstate)
    context().update(optax.apply_updates(params, updates))
    return {'loss': loss.mean(), 'grad_norm': optax.global_norm(grads)}
