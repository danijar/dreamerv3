import pathlib
import sys
from functools import partial as bind

sys.path.append(str(pathlib.Path(__file__).parent.parent.parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent))

import embodied
import numpy as np
import pytest

import utils


class TestTrain:

  @pytest.mark.parametrize('strategy', ('blocking', 'process', 'thread'))
  def test_run_loop(self, tmpdir, strategy):
    args = self._make_args(tmpdir)
    agent = self._make_agent()
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, self._make_logger, args)
    stats = agent.stats()
    print('Stats:', stats)
    replay_steps = args.steps * args.train_ratio
    assert stats['lifetime'] > 8  # Otherwise decrease log and ckpt interval.
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)
    assert np.allclose(stats['replay_steps'], replay_steps, 100, 0.1)
    assert stats['reports'] >= 1
    assert stats['saves'] >= 2
    assert stats['loads'] == 0
    args = args.update(steps=args.steps + 1e4)
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, self._make_logger, args)
    stats = agent.stats()
    assert stats['loads'] == 1
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)

  def _make_agent(self):
    env = self._make_env(0)
    agent = utils.TestAgent(env.obs_space, env.act_space)
    env.close()
    return agent

  def _make_env(self, index):
    from embodied.envs import dummy
    return dummy.Dummy('disc', size=(64, 64), length=100)

  def _make_replay(self, args):
    kwargs = {'length': args.batch_length, 'capacity': 1e4}
    return embodied.replay.Replay(**kwargs)

  def _make_logger(self):
    return embodied.Logger(embodied.Counter(), [
        embodied.logger.TerminalOutput(),
    ])

  def _make_args(self, logdir):
    return embodied.Config(
        logdir=str(logdir),
        num_envs=4,
        steps=5e4,
        log_every=3,
        save_every=5,
        eval_every=5,
        train_ratio=32.0,
        train_fill=100,
        batch_size=8,
        batch_length=16,
        batch_length_eval=8,
        replay_context=0,
        expl_until=0,
        from_checkpoint='',
        usage=dict(psutil=True, nvsmi=False),
        log_zeros=False,
        log_video_streams=4,
        log_video_fps=20,
        log_keys_video=['image'],
        log_keys_sum='^$',
        log_keys_avg='^$',
        log_keys_max='^$',
        driver_parallel=True,
    )
