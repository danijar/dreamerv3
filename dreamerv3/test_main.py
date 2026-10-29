import elements
import pytest
import ruamel.yaml as yaml

from dreamerv3 import main


def make_config(logdir, **updates):
  configs = yaml.YAML(typ='safe').load(
      (elements.Path(main.__file__).parent / 'configs.yaml').read())
  # Without the online queue, every sample is drawn by the replay selector.
  return elements.Config(configs['defaults']).update({
      'logdir': str(logdir), 'batch_size': 1, 'batch_length': 4,
      'replay_context': 0, 'replay.size': 1000, 'replay.online': False,
      **updates})


def sampled_steps(config):
  replay = main.make_replay(config, 'replay')
  for step in range(100):
    replay.add({'step': step})
  return [int(replay.sample(1)['step'][0, 0]) for _ in range(8)]


@pytest.mark.parametrize('first, second', [
    (dict(seed=0), dict(seed=1)),
    (dict(replicas=2, replica=0), dict(replicas=2, replica=1)),
])
def test_replay_draws_differ_across_seeds_and_replicas(
    tmp_path, first, second):
  assert (sampled_steps(make_config(tmp_path / 'a', **first)) !=
          sampled_steps(make_config(tmp_path / 'b', **second)))


def test_replay_draws_repeat_for_the_same_seed(tmp_path):
  assert (sampled_steps(make_config(tmp_path / 'a', seed=3)) ==
          sampled_steps(make_config(tmp_path / 'b', seed=3)))
