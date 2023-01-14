from minerl.herobraine.env_spec import EnvSpec
from minerl.herobraine.hero import handler
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero import mc
from minerl.herobraine.hero.mc import INVERSE_KEYMAP


def edit_options(**kwargs):
  import os, pathlib, re
  for word in os.popen('pip3 --version').read().split(' '):
    if '-packages/pip' in word:
      break
  else:
    raise RuntimeError('Could not found python package directory.')
  packages = pathlib.Path(word).parent
  filename = packages / 'minerl/Malmo/Minecraft/run/options.txt'
  options = filename.read_text()
  if 'fovEffectScale:' not in options:
    options += 'fovEffectScale:1.0\n'
  if 'simulationDistance:' not in options:
    options += 'simulationDistance:12\n'
  for key, value in kwargs.items():
    assert f'{key}:' in options, key
    assert isinstance(value, str), (value, type(value))
    options = re.sub(f'{key}:.*\n', f'{key}:{value}\n', options)
  filename.write_text(options)


edit_options(
    difficulty='2',
    renderDistance='6',
    simulationDistance='6',
    fovEffectScale='0.0',
    ao='1',
    gamma='5.0',
)


class MineRLEnv(EnvSpec):

  def __init__(self, resolution=(64, 64), break_speed=50, gamma=10.0):
    self.resolution = resolution
    self.break_speed = break_speed
    self.gamma = gamma
    super().__init__(name='MineRLEnv-v1')

  def create_agent_start(self):
    return [
        BreakSpeedMultiplier(self.break_speed),
    ]

  def create_agent_handlers(self):
    return []

  def create_server_world_generators(self):
    return [handlers.DefaultWorldGenerator(force_reset=True)]

  def create_server_quit_producers(self):
    return [handlers.ServerQuitWhenAnyAgentFinishes()]

  def create_server_initial_conditions(self):
    return [
        handlers.TimeInitialCondition(
            allow_passage_of_time=True,
            start_time=0,
        ),
        handlers.SpawningInitialCondition(
            allow_spawning=True,
        )
    ]

  def create_observables(self):
    return [
        handlers.POVObservation(self.resolution),
        handlers.FlatInventoryObservation(mc.ALL_ITEMS),
        handlers.EquippedItemObservation(
            mc.ALL_ITEMS, _default='air', _other='other'),
        handlers.ObservationFromCurrentLocation(),
        handlers.ObservationFromLifeStats(),
    ]

  def create_actionables(self):
    kw = dict(_other='none', _default='none')
    return [
        handlers.KeybasedCommandAction('forward', INVERSE_KEYMAP['forward']),
        handlers.KeybasedCommandAction('back', INVERSE_KEYMAP['back']),
        handlers.KeybasedCommandAction('left', INVERSE_KEYMAP['left']),
        handlers.KeybasedCommandAction('right', INVERSE_KEYMAP['right']),
        handlers.KeybasedCommandAction('jump', INVERSE_KEYMAP['jump']),
        handlers.KeybasedCommandAction('sneak', INVERSE_KEYMAP['sneak']),
        handlers.KeybasedCommandAction('attack', INVERSE_KEYMAP['attack']),
        handlers.CameraAction(),
        handlers.PlaceBlock(['none'] + mc.ALL_ITEMS, **kw),
        handlers.EquipAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.CraftNearbyAction(['none'] + mc.ALL_ITEMS, **kw),
        handlers.SmeltItemNearby(['none'] + mc.ALL_ITEMS, **kw),
    ]

  def is_from_folder(self, folder):
    return folder == 'none'

  def get_docstring(self):
    return ''

  def determine_success_from_rewards(self, rewards):
    return True

  def create_rewardables(self):
    return []

  def create_server_decorators(self):
    return []

  def create_mission_handlers(self):
    return []

  def create_monitors(self):
    return []


class BreakSpeedMultiplier(handler.Handler):

  def __init__(self, multiplier=1.0):
    self.multiplier = multiplier

  def to_string(self):
    return f'break_speed({self.multiplier})'

  def xml_template(self):
    return '<BreakSpeedMultiplier>{{multiplier}}</BreakSpeedMultiplier>'


class Gamma(handler.Handler):

  def __init__(self, gamma=2.0):
    self.gamma = gamma

  def to_string(self):
    return f'gamma({self.gamma})'

  def xml_template(self):
    return '<GammaSetting>{{gamma}}</GammaSetting>'


NOOP_ACTION = dict(
    camera=(0, 0), forward=0, back=0, left=0, right=0, attack=0, sprint=0,
    jump=0, sneak=0, craft='none', nearbyCraft='none', nearbySmelt='none',
    place='none', equip='none',
)
