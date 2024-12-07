from functools import partial as bind

import elements
import embodied
import numpy as np

import utils


class TestTrain:

  def test_run_loop(self, tmpdir):
    args = self._make_args(tmpdir)
    agent = self._make_agent()
    embodied.run.train(
        lambda: agent, bind(self._make_replay, args),
        self._make_env, self._make_logger, args)
    stats = agent.stats()
    print('Stats:', stats)
    replay_steps = args.steps * args.train_ratio
    assert stats['lifetime'] >= 1  # Otherwise decrease log and ckpt interval.
    assert np.allclose(stats['env_steps'], args.steps, 100, 0.1)
    assert np.allclose(stats['replay_steps'], replay_steps, 100, 0.1)
    assert stats['reports'] >= 1
    assert stats['saves'] >= 2
    assert stats['loads'] == 0
    args = args.update(steps=2 * args.steps)
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
    return elements.Logger(elements.Counter(), [
        elements.logger.TerminalOutput(),
    ])

  def _make_args(self, logdir):
    return elements.Config(
        steps=1000,
        train_ratio=32.0,
        log_every=0.1,
        report_every=0.2,
        save_every=0.2,
        report_batches=1,
        from_checkpoint='',
        usage=dict(psutil=True),
        debug=False,
        logdir=str(logdir),
        envs=4,
        batch_size=8,
        batch_length=16,
        replay_context=0,
        report_length=8,
    )
