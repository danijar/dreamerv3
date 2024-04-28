import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import embodied
import numpy as np


class TestDriver:

  def test_episode_length(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=1)
    assert len(seq) == 11

  def test_first_step(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [0, 11]:
      assert seq[index]['is_first'].item() is True
      assert seq[index]['is_last'].item() is False
    for index in [1, 10, 12]:
      assert seq[index]['is_first'].item() is False

  def test_last_step(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(agent.policy, episodes=2)
    for index in [10, 21]:
      assert seq[index]['is_last'].item() is True
      assert seq[index]['is_first'].item() is False
    for index in [0, 1, 9, 11, 20]:
      assert seq[index]['is_last'].item() is False

  def test_env_reset(self):
    agent = self._make_agent()
    driver = embodied.Driver([bind(self._make_env, length=5)])
    driver.reset(agent.init_policy)
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    action = np.array([1])
    driver(lambda obs, state: ({'action': action}, {}, state), episodes=2)
    assert len(seq) == 12
    seq = {k: np.array([seq[i][k] for i in range(len(seq))]) for k in seq[0]}
    assert (seq['is_first'] == [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]).all()
    assert (seq['is_last']  == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (seq['reset']    == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]).all()
    assert (seq['action']   == [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]).all()

  def test_agent_inputs(self):
    agent = self._make_agent()
    driver = embodied.Driver([self._make_env])
    driver.reset(agent.init_policy)
    inputs = []
    states = []
    def policy(obs, state=None, mode='train'):
      inputs.append(obs)
      states.append(state)
      act, _, _ = agent.policy(obs, state, mode)
      return act, {}, 'state'
    seq = []
    driver.on_step(lambda tran, _: seq.append(tran))
    driver(policy, episodes=2)
    assert len(seq) == 22
    assert states == ([()] + ['state'] * 21)
    for index in [0, 11]:
      assert inputs[index]['is_first'].item() is True
    for index in [1, 10, 12, 21]:
      assert inputs[index]['is_first'].item() is False
    for index in [10, 21]:
      assert inputs[index]['is_last'].item() is True
    for index in [0, 1, 9, 11, 20]:
      assert inputs[index]['is_last'].item() is False

  def test_unexpected_reset(self):

    class UnexpectedReset(embodied.Wrapper):
      """Send is_first without preceeding is_last."""
      def __init__(self, env, when):
        super().__init__(env)
        self._when = when
        self._step = 0
      def step(self, action):
        if self._step == self._when:
          action = action.copy()
          action['reset'] = np.ones_like(action['reset'])
        self._step += 1
        return self.env.step(action)

    env = self._make_env(length=4)
    env = UnexpectedReset(env, when=3)
    agent = self._make_agent()
    driver = embodied.Driver([lambda: env])
    driver.reset(agent.init_policy)
    steps = []
    driver.on_step(lambda tran, _: steps.append(tran))
    driver(agent.policy, episodes=1)
    assert len(steps) == 8
    steps = {k: np.array([x[k] for x in steps]) for k in steps[0]}
    assert (steps['reset'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()
    assert (steps['is_first'] == [1, 0, 0, 1, 0, 0, 0, 0]).all()
    assert (steps['is_last'] == [0, 0, 0, 0, 0, 0, 0, 1]).all()

  def _make_env(self, length=10):
    from embodied.envs import dummy
    return dummy.Dummy('disc', length=length)

  def _make_agent(self):
    env = self._make_env()
    agent = embodied.RandomAgent(env.obs_space, env.act_space)
    env.close()
    return agent
