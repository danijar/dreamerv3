class Agent:

  def __init__(self, obs_space, act_space, config):
    pass

  def init_train(self, batch_size):
    raise NotImplementedError('init_train(batch_size) -> carry')

  def init_report(self, batch_size):
    raise NotImplementedError('init_report(batch_size) -> carry')

  def init_policy(self, batch_size):
    raise NotImplementedError('init_policy(batch_size) -> carry')

  def train(self, carry, data):
    raise NotImplementedError('train(carry, data) -> carry, out, metrics')

  def report(self, carry, data):
    raise NotImplementedError('report(carry, data) -> carry, metrics')

  def policy(self, carry, obs, mode):
    raise NotImplementedError('policy(carry, obs, mode) -> carry, act, out')

  def stream(self, st):
    raise NotImplementedError('stream(st) -> st')

  def save(self):
    raise NotImplementedError('save() -> data')

  def load(self, data):
    raise NotImplementedError('load(data) -> None')


class Env:

  def __repr__(self):
    return (
        f'{self.__class__.__name__}('
        f'obs_space={self.obs_space}, '
        f'act_space={self.act_space})')

  @property
  def obs_space(self):
    # The observation space must contain the keys is_first, is_last, and
    # is_terminal. Commonly, it also contains the keys reward and image. By
    # convention, keys starting with 'log/' are not consumed by the agent.
    raise NotImplementedError('Returns: dict of spaces')

  @property
  def act_space(self):
    # The action space must contain the reset key as well as any actions.
    raise NotImplementedError('Returns: dict of spaces')

  def step(self, action):
    raise NotImplementedError('Returns: dict')

  def close(self):
    pass


class Stream:

  def __iter__(self):
    return self

  def __next__(self):
    raise NotImplementedError

  def save(self):
    raise NotImplementedError

  def load(self, state):
    raise NotImplementedError
