import time

import cloudpickle
import numpy as np

from .. import distr


class Driver:

  def __init__(self, make_env_fns, parallel=True, **kwargs):
    assert len(make_env_fns) >= 1
    self.parallel = parallel
    self.kwargs = kwargs
    self.length = len(make_env_fns)
    if parallel:
      import multiprocessing as mp
      context = mp.get_context()
      self.pipes, pipes = zip(*[context.Pipe() for _ in range(self.length)])
      fns = [cloudpickle.dumps(fn) for fn in make_env_fns]
      self.procs = [
          distr.StoppableProcess(self._env_server, i, pipe, fn, start=True)
          for i, (fn, pipe) in enumerate(zip(fns, pipes))]
      self.pipes[0].send(('act_space',))
      self.act_space = self._receive(self.pipes[0])
    else:
      self.envs = [fn() for fn in make_env_fns]
      self.act_space = self.envs[0].act_space
    self.callbacks = []
    self.acts = None
    self.carry = None
    self.reset()

  def reset(self, init_policy=None):
    self.acts = {
        k: np.zeros((self.length,) + v.shape, v.dtype)
        for k, v in self.act_space.items()}
    self.acts['reset'] = np.ones(self.length, bool)
    self.carry = init_policy and init_policy(self.length)

  def close(self):
    if self.parallel:
      [proc.stop() for proc in self.procs]
    else:
      [env.close() for env in self.envs]

  def on_step(self, callback):
    self.callbacks.append(callback)

  def __call__(self, policy, steps=0, episodes=0):
    step, episode = 0, 0
    while step < steps or episode < episodes:
      step, episode = self._step(policy, step, episode)

  def _step(self, policy, step, episode):
    acts = self.acts
    assert all(len(x) == self.length for x in acts.values())
    assert all(isinstance(v, np.ndarray) for v in acts.values())
    acts = [{k: v[i] for k, v in acts.items()} for i in range(self.length)]
    if self.parallel:
      [pipe.send(('step', act)) for pipe, act in zip(self.pipes, acts)]
      obs = [self._receive(pipe) for pipe in self.pipes]
    else:
      obs = [env.step(act) for env, act in zip(self.envs, acts)]
    obs = {k: np.stack([x[k] for x in obs]) for k in obs[0].keys()}
    assert all(len(x) == self.length for x in obs.values()), obs
    acts, outs, self.carry = policy(obs, self.carry, **self.kwargs)
    assert all(k not in acts for k in outs), (
        list(outs.keys()), list(acts.keys()))
    if obs['is_last'].any():
      mask = ~obs['is_last']
      acts = {k: self._mask(v, mask) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self.acts = acts
    trans = {**obs, **acts, **outs}
    for i in range(self.length):
      trn = {k: v[i] for k, v in trans.items()}
      [fn(trn, i, **self.kwargs) for fn in self.callbacks]
    step += len(obs['is_first'])
    episode += obs['is_last'].sum()
    return step, episode

  def _mask(self, value, mask):
    while mask.ndim < value.ndim:
      mask = mask[..., None]
    return value * mask.astype(value.dtype)

  def _receive(self, pipe):
    try:
      msg, arg = pipe.recv()
      if msg == 'error':
        raise RuntimeError(arg)
      assert msg == 'result'
      return arg
    except Exception:
      print('Terminating workers due to an exception.')
      [proc.kill() for proc in self.procs]
      raise

  @staticmethod
  def _env_server(context, envid, pipe, ctor):
    try:
      ctor = cloudpickle.loads(ctor)
      env = ctor()
      while context.running:
        if not pipe.poll(0.1):
          time.sleep(0.1)
          continue
        try:
          msg, *args = pipe.recv()
        except EOFError:
          return
        if msg == 'step':
          assert len(args) == 1
          act = args[0]
          obs = env.step(act)
          pipe.send(('result', obs))
        elif msg == 'obs_space':
          assert len(args) == 0
          pipe.send(('result', env.obs_space))
        elif msg == 'act_space':
          assert len(args) == 0
          pipe.send(('result', env.act_space))
        else:
          raise ValueError(f'Invalid message {msg}')
    except Exception as e:
      distr.warn_remote_error(e, f'Env{envid}')
      pipe.send(('error', e))
    finally:
      print(f'Closing env {envid}')
      env.close()
      pipe.close()
