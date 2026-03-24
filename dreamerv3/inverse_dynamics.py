import argparse
import pathlib
from dataclasses import dataclass
from functools import partial

import elements
import embodied
import jax
import jax.numpy as jnp
import numpy as np
import optax
import ruamel.yaml as yaml

from . import main as dv3_main
import nle.dataset as nld
from PIL import Image

# -----------------------------
# Config / loading helpers
# -----------------------------

def load_saved_config(logdir):
    logdir = elements.Path(logdir)
    path = logdir / 'config.yaml'
    if not path.exists():
        raise FileNotFoundError(f'Could not find config at: {path}')
    data = yaml.YAML(typ='safe').load(path.read())
    return elements.Config(data)


def make_replay_from_config(config, replay_dir):
    batlen = config.batch_length
    consec = config.consec_train
    capacity = int(config.replay.size)
    length = consec * batlen + config.replay_context

    replay = embodied.replay.Replay(
        length=length,
        capacity=capacity,
        online=config.replay.online,
        chunksize=config.replay.chunksize,
        directory=elements.Path(replay_dir),
        name='inverse_dynamics_replay',
        seed=config.seed,
    )
    replay.load()
    return replay


def make_env_spaces(config):
    env = dv3_main.make_env(config, 0)
    obs_space = {k: v for k, v in env.obs_space.items() if not k.startswith('log/')}
    act_space = {k: v for k, v in env.act_space.items() if k != 'reset'}
    env.close()
    return obs_space, act_space


# -----------------------------
# Model
# -----------------------------

@dataclass
class InvDynConfig:
    hidden: int = 512
    hidden2: int = 512
    lr: float = 1e-4
    batch_size: int = 16
    train_steps: int = 10_000
    log_every: int = 100
    eval_every: int = 500
    seed: int = 0
    include_reward: bool = True


def init_mlp_params(rng, in_dim, hidden, hidden2, out_dim):
    k1, k2, k3 = jax.random.split(rng, 3)
    scale = 0.02
    params = {
        'w1': scale * jax.random.normal(k1, (in_dim, hidden)),
        'b1': jnp.zeros((hidden,)),
        'w2': scale * jax.random.normal(k2, (hidden, hidden2)),
        'b2': jnp.zeros((hidden2,)),
        'w3': scale * jax.random.normal(k3, (hidden2, out_dim)),
        'b3': jnp.zeros((out_dim,)),
    }
    return params


def mlp_apply(params, x):
    x = x @ params['w1'] + params['b1']
    x = jax.nn.silu(x)
    x = x @ params['w2'] + params['b2']
    x = jax.nn.silu(x)
    x = x @ params['w3'] + params['b3']
    return x


# -----------------------------
# Observation featurization
# -----------------------------

def _flatten_obs_part(x):
    x = np.asarray(x)
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255.0
    else:
        x = x.astype(np.float32)
    return x.reshape((x.shape[0], -1))


def _pair_to_input(obs_t, obs_tp1, reward_t, include_reward=True):
    """
    Image-only inverse dynamics input:
      x = [flatten(image_t), flatten(image_t+1), reward_t]
    """
    if 'image' not in obs_t or 'image' not in obs_tp1:
        raise KeyError(
            f"Expected 'image' key in obs_t/obs_tp1, got keys "
            f"{list(obs_t.keys())} and {list(obs_tp1.keys())}"
        )

    parts = [
        _flatten_obs_part(obs_t['image']),
        _flatten_obs_part(obs_tp1['image']),
    ]
    if include_reward:
        parts.append(np.asarray(reward_t, np.float32).reshape((-1, 1)))
    x = np.concatenate(parts, axis=-1)
    return x.astype(np.float32)


def _extract_transitions(batch):
    """
    batch keys are expected to be [B, T, ...]
    Returns flattened transition dataset:
      obs_t, obs_tp1, reward_t, action_t
    """
    obs_keys = [k for k in batch.keys() if k not in ('action', 'consec', 'stepid')]
    obs_t = {k: batch[k][:, :-1] for k in obs_keys}
    obs_tp1 = {k: batch[k][:, 1:] for k in obs_keys}
    reward_t = batch['reward'][:, :-1]

    # action may be scalar discrete or already shaped
    action_t = batch['action'][:, :-1]

    def flatten_bt(x):
        x = np.asarray(x)
        b, t = x.shape[:2]
        return x.reshape((b * t,) + x.shape[2:])

    obs_t = {k: flatten_bt(v) for k, v in obs_t.items()}
    obs_tp1 = {k: flatten_bt(v) for k, v in obs_tp1.items()}
    reward_t = flatten_bt(reward_t).reshape(-1)
    action_t = flatten_bt(action_t)

    return obs_t, obs_tp1, reward_t, action_t


# -----------------------------
# Training
# -----------------------------

def _infer_num_actions(act_space):
    if 'action' not in act_space:
        raise KeyError(f"Expected discrete action key 'action', got: {list(act_space.keys())}")
    space = act_space['action']
    if not space.discrete:
        raise ValueError("This inverse dynamics file currently supports discrete 'action' only.")
    return int(space.high)


@partial(jax.jit, static_argnums=(3,))
def train_step(params, opt_state, batch, include_reward):
    x, y = batch

    def loss_fn(p):
        logits = mlp_apply(p, x)
        labels = y.astype(jnp.int32).reshape(-1)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        pred = jnp.argmax(logits, axis=-1)
        acc = (pred == labels).mean()
        top5 = jnp.any(jnp.argsort(logits, axis=-1)[:, -5:] == labels[:, None], axis=-1).mean()
        return loss, {'acc': acc, 'top5': top5}

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    metrics = {'loss': loss, **metrics}
    return params, opt_state, metrics


@jax.jit
def eval_step(params, batch):
    x, y = batch
    logits = mlp_apply(params, x)
    labels = y.astype(jnp.int32).reshape(-1)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    pred = jnp.argmax(logits, axis=-1)
    acc = (pred == labels).mean()
    top5 = jnp.any(jnp.argsort(logits, axis=-1)[:, -5:] == labels[:, None], axis=-1).mean()
    return {'loss': loss, 'acc': acc, 'top5': top5}


class InverseDynamicsTrainer:

    def __init__(self, config, replay_dir, invcfg: InvDynConfig):
        self.config = config
        self.replay_dir = elements.Path(replay_dir)
        self.invcfg = invcfg

        self.obs_space, self.act_space = make_env_spaces(config)
        self.num_actions = _infer_num_actions(self.act_space)

        self.replay = make_replay_from_config(config, replay_dir)
        self.stream = iter(dv3_main.make_stream(config, self.replay, 'train'))

        # Infer input dimension from one real batch
        probe = next(self.stream)
        obs_t, obs_tp1, reward_t, action_t = _extract_transitions(probe)
        x_probe = _pair_to_input(obs_t, obs_tp1, reward_t, include_reward=invcfg.include_reward)
        self.input_dim = x_probe.shape[-1]

        rng = jax.random.PRNGKey(invcfg.seed)
        self.params = init_mlp_params(
            rng, self.input_dim, invcfg.hidden, invcfg.hidden2, self.num_actions)
        global opt
        opt = optax.adam(invcfg.lr)
        self.opt_state = opt.init(self.params)

    def _sample_numpy_batch(self):
        batch = next(self.stream)
        obs_t, obs_tp1, reward_t, action_t = _extract_transitions(batch)
        x = _pair_to_input(obs_t, obs_tp1, reward_t, include_reward=self.invcfg.include_reward)
        y = np.asarray(action_t).reshape(-1)
        return x, y

    def train(self, steps=None):
        steps = steps or self.invcfg.train_steps
        history = []

        for step in range(1, steps + 1):
            x, y = self._sample_numpy_batch()
            x = jnp.asarray(x)
            y = jnp.asarray(y)
            self.params, self.opt_state, mets = train_step(
                self.params, self.opt_state, (x, y), self.invcfg.include_reward)

            if step % self.invcfg.log_every == 0 or step == 1:
                m = jax.tree.map(lambda v: float(v), mets)
                history.append((step, m))
                print(
                    f"[inv] step={step:6d} "
                    f"loss={m['loss']:.4f} "
                    f"acc={m['acc']:.4f} "
                    f"top5={m['top5']:.4f}"
                )

        return history

    def evaluate_once(self):
        x, y = self._sample_numpy_batch()
        mets = eval_step(self.params, (jnp.asarray(x), jnp.asarray(y)))
        return jax.tree.map(lambda v: float(v), mets)

    def save(self, outpath):
        outpath = elements.Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(outpath),
            **jax.tree.map(np.array, self.params),
            input_dim=np.array(self.input_dim),
            num_actions=np.array(self.num_actions),
        )
        print(f"Saved inverse dynamics params to {outpath}")


def train_inverse_dynamics(
    logdir,
    replay_dir=None,
    steps=10_000,
    lr=1e-4,
    hidden=512,
    hidden2=512,
    include_reward=True,
    seed=0,
    save_name='inverse_dynamics_params.npz',
):
    logdir = elements.Path(logdir)
    replay_dir = elements.Path(replay_dir) if replay_dir else (logdir / 'replay')

    config = load_saved_config(logdir)

    invcfg = InvDynConfig(
        hidden=hidden,
        hidden2=hidden2,
        lr=lr,
        train_steps=steps,
        include_reward=include_reward,
        seed=seed,
    )

    trainer = InverseDynamicsTrainer(config, replay_dir, invcfg)
    trainer.train(steps)

    eval_mets = trainer.evaluate_once()
    print(f"[inv] eval loss={eval_mets['loss']:.4f} acc={eval_mets['acc']:.4f} top5={eval_mets['top5']:.4f}")

    trainer.save(logdir / save_name)
    return trainer

def tty_to_rgb(chars, colors, size=(64, 64)):
    chars = np.asarray(chars, dtype=np.uint8)
    colors = np.asarray(colors, dtype=np.uint8)

    rgb = np.zeros((chars.shape[0], chars.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = chars
    rgb[..., 1] = (colors.astype(np.int32) * 16).clip(0, 255).astype(np.uint8)
    rgb[..., 2] = ((chars.astype(np.int32) // 2) + (colors.astype(np.int32) * 8)).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(rgb)
    img = img.resize(size, Image.BILINEAR)
    return np.array(img, dtype=np.uint8)
def batch_to_pairs_nld_aa(mb):
    images = []

    B, T = mb["tty_chars"].shape[:2]
    assert B == 1, f"Expected batch_size=1 for now, got B={B}"

    for t in range(T):
        img = tty_to_rgb(mb["tty_chars"][0, t], mb["tty_colors"][0, t])
        images.append(img)

    images = np.stack(images, axis=0)  # [T, 64, 64, 3]
    actions = np.asarray(mb["keypresses"][0], dtype=np.int32)
    scores = np.asarray(mb["scores"][0], dtype=np.float32)
    rewards = np.diff(scores, prepend=scores[:1])

    obs_t = {"image": images[:-1]}
    obs_tp1 = {"image": images[1:]}
    reward_t = rewards[:-1]
    action_t = actions[:-1]

    return obs_t, obs_tp1, reward_t, action_t
def make_nld_aa_dataset(dataset_name, batch_size=1):
    return nld.TtyrecDataset(dataset_name, batch_size=batch_size)
class NLDInverseDynamicsTrainer:

    def __init__(self, dataset_name, invcfg: InvDynConfig, include_reward=True):
        self.dataset_name = dataset_name
        self.invcfg = invcfg
        self.include_reward = include_reward

        self.dataset = make_nld_aa_dataset(dataset_name, batch_size=1)

        probe = next(iter(self.dataset))
        obs_t, obs_tp1, reward_t, action_t = batch_to_pairs_nld_aa(probe)
        x_probe = _pair_to_input(obs_t, obs_tp1, reward_t, include_reward=include_reward)

        self.input_dim = x_probe.shape[-1]
        self.num_actions = int(np.max(action_t)) + 1

        rng = jax.random.PRNGKey(invcfg.seed)
        self.params = init_mlp_params(
            rng, self.input_dim, invcfg.hidden, invcfg.hidden2, self.num_actions
        )

        global opt
        opt = optax.adam(invcfg.lr)
        self.opt_state = opt.init(self.params)

    def _sample_numpy_batch(self):
        mb = next(iter(self.dataset))
        obs_t, obs_tp1, reward_t, action_t = batch_to_pairs_nld_aa(mb)
        x = _pair_to_input(obs_t, obs_tp1, reward_t, include_reward=self.include_reward)
        y = np.asarray(action_t, dtype=np.int32).reshape(-1)
        return x, y

    def train(self, steps=None):
        steps = steps or self.invcfg.train_steps
        history = []

        for step in range(1, steps + 1):
            x, y = self._sample_numpy_batch()
            x = jnp.asarray(x)
            y = jnp.asarray(y)

            self.params, self.opt_state, mets = train_step(
                self.params, self.opt_state, (x, y), self.include_reward
            )

            if step % self.invcfg.log_every == 0 or step == 1:
                m = jax.tree.map(lambda v: float(v), mets)
                history.append((step, m))
                print(
                    f"[nld-aa inv] step={step:6d} "
                    f"loss={m['loss']:.4f} "
                    f"acc={m['acc']:.4f} "
                    f"top5={m['top5']:.4f}"
                )

        return history

    def evaluate_once(self):
        x, y = self._sample_numpy_batch()
        mets = eval_step(self.params, (jnp.asarray(x), jnp.asarray(y)))
        return jax.tree.map(lambda v: float(v), mets)

    def save(self, outpath):
        outpath = elements.Path(outpath)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(outpath),
            **jax.tree.map(np.array, self.params),
            input_dim=np.array(self.input_dim),
            num_actions=np.array(self.num_actions),
        )
        print(f"Saved inverse dynamics params to {outpath}")
def train_inverse_dynamics_nld_aa(
    dataset_name="nld-aa-v0",
    steps=10_000,
    lr=1e-4,
    hidden=512,
    hidden2=512,
    include_reward=True,
    seed=0,
    save_path="/content/drive/MyDrive/nld_aa_inverse_image_only.npz",
):
    invcfg = InvDynConfig(
        hidden=hidden,
        hidden2=hidden2,
        lr=lr,
        train_steps=steps,
        include_reward=include_reward,
        seed=seed,
    )

    trainer = NLDInverseDynamicsTrainer(
        dataset_name=dataset_name,
        invcfg=invcfg,
        include_reward=include_reward,
    )
    trainer.train(steps)

    eval_mets = trainer.evaluate_once()
    print(
        f"[nld-aa inv] eval "
        f"loss={eval_mets['loss']:.4f} "
        f"acc={eval_mets['acc']:.4f} "
        f"top5={eval_mets['top5']:.4f}"
    )

    trainer.save(save_path)
    return trainer
def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, required=True)
    parser.add_argument('--replay_dir', type=str, default=None)
    parser.add_argument('--steps', type=int, default=10_000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--hidden2', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--no_reward', action='store_true')
    parser.add_argument('--save_name', type=str, default='inverse_dynamics_params.npz')
    args = parser.parse_args()
    parser.add_argument('--source', type=str, default='replay', choices=['replay', 'nld-aa'])
    parser.add_argument('--dataset_name', type=str, default='nld-aa-v0')
    parser.add_argument('--save_path', type=str, default=None)

    if args.source == 'replay':
        train_inverse_dynamics(
            logdir=args.logdir,
            replay_dir=args.replay_dir,
            steps=args.steps,
            lr=args.lr,
            hidden=args.hidden,
            hidden2=args.hidden2,
            include_reward=not args.no_reward,
            seed=args.seed,
            save_name=args.save_name,
        )
    else:
        save_path = args.save_path
        if save_path is None:
            save_path = f"/content/drive/MyDrive/{args.dataset_name}_inverse_image_only.npz"

        train_inverse_dynamics_nld_aa(
            dataset_name=args.dataset_name,
            steps=args.steps,
            lr=args.lr,
            hidden=args.hidden,
            hidden2=args.hidden2,
            include_reward=not args.no_reward,
            seed=args.seed,
            save_path=save_path,
        )


if __name__ == '__main__':
    main_cli()