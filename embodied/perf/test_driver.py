import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

import elements
import embodied


class TestDriver:

  def test_throughput_dummy(self, parallel=True):
    from embodied.envs import dummy
    make_env_fns = [bind(dummy.Dummy, 'disc') for _ in range(32)]
    example = make_env_fns[0]()
    agent = embodied.RandomAgent(example.obs_space, example.act_space)
    example.close()
    driver = embodied.Driver(make_env_fns, parallel)
    driver.reset(agent.init_policy)
    fps = elements.FPS()
    while True:
      driver(agent.policy, steps=100)
      fps.step(100 * len(make_env_fns))
      print(f'FPS: {fps.result():.0f}')

  def test_throughput_crafter(self, parallel=True):
    from embodied.envs import crafter
    make_env_fns = [bind(crafter.Crafter, 'reward') for _ in range(32)]
    example = make_env_fns[0]()
    agent = embodied.RandomAgent(example.obs_space, example.act_space)
    example.close()
    driver = embodied.Driver(make_env_fns, parallel)
    driver.reset(agent.init_policy)
    fps = elements.FPS()
    while True:
      driver(agent.policy, steps=100)
      fps.step(100 * len(make_env_fns))
      print(f'FPS: {fps.result():.0f}')
