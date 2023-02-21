# Mastering Diverse Domains through World Models

A reimplementation of [DreamerV3][paper], a scalable and general reinforcement
learning algorithm that masters a wide range of applications with fixed
hyperparameters.

![DreamerV3 Tasks](https://user-images.githubusercontent.com/2111293/217647148-cbc522e2-61ad-4553-8e14-1ecdc8d9438b.gif)

If you find this code useful, please reference in your paper:

```
@article{hafner2023dreamerv3,
  title={Mastering Diverse Domains through World Models},
  author={Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2301.04104},
  year={2023}
}
```

To learn more:

- [Research paper][paper]
- [Project website][website]
- [Twitter summary][tweet]

## DreamerV3

DreamerV3 learns a world model from experiences and uses it to train an actor
critic policy from imagined trajectories. The world model encodes sensory
inputs into categorical representations and predicts future representations and
rewards given actions.

![DreamerV3 Method Diagram](https://user-images.githubusercontent.com/2111293/217355673-4abc0ce5-1a4b-4366-a08d-64754289d659.png)

DreamerV3 masters a wide range of domains with a fixed set of hyperparameters,
outperforming specialized methods. Removing the need for tuning reduces the
amount of expert knowledge and computational resources needed to apply
reinforcement learning.

![DreamerV3 Benchmark Scores](https://user-images.githubusercontent.com/2111293/217356042-536a693a-cb5e-42aa-a20f-5303a77cad9c.png)

Due to its robustness, DreamerV3 shows favorable scaling properties. Notably,
using larger models consistently increases not only its final performance but
also its data-efficiency. Increasing the number of gradient steps further
increases data efficiency.

![DreamerV3 Scaling Behavior](https://user-images.githubusercontent.com/2111293/217356063-0cf06b17-89f0-4d5f-85a9-b583438c98dd.png)

# Instructions

## Package

If you just want to run DreamerV3 on a custom environment, you can `pip install
dreamerv3` and copy [`example.py`][example] from this repository as a starting
point.

## Docker

If you want to make modifications to the code, you can either use the provided
`Dockerfile` that contains instructions or follow the manual instructions
below.

## Manual

Install [JAX][jax] and then the other dependencies:

```sh
pip install -r requirements.txt
```

Simple training script:

```sh
python example.py
```

Flexible training script:

```sh
python dreamerv3/train.py \
  --logdir ~/logdir/$(date "+%Y%m%d-%H%M%S") \
  --configs crafter --batch_size 16 --run.train_ratio 32
```

# Tips

- All config options are listed in `configs.yaml` and you can override them
  from the command line.
- The `debug` config block reduces the network size, batch size, duration
  between logs, and so on for fast debugging (but does not learn a good model).
- By default, the code tries to run on GPU. You can switch to CPU or TPU using
  the `--jax.platform cpu` flag. Note that multi-GPU support is untested.
- You can run with multiple config blocks that will override defaults in the
  order they are specified, for example `--configs crafter large`.
- By default, metrics are printed to the terminal, appended to a JSON lines
  file, and written as TensorBoard summaries. Other outputs like WandB can be
  enabled in the training script.
- If you get a `Too many leaves for PyTreeDef` error, it means you're
  reloading a checkpoint that is not compatible with the current config. This
  often happens when reusing an old logdir by accident.
- If you are getting CUDA errors, scroll up because the cause is often just an
  error that happened earlier, such as out of memory or incompatible JAX and
  CUDA versions.
- You can use the `small`, `medium`, `large` config blocks to reduce memory
  requirements. The default is `xlarge`. See the scaling graph above to see how
  this affects performance.
- Many environments are included, some of which require installating additional
  packages. See the installation scripts in `scripts` and the `Dockerfile` for
  reference.
- When running on custom environments, make sure to specify the observation
  keys the agent should be using via `encoder.mlp_keys`, `encode.cnn_keys`,
  `decoder.mlp_keys` and `decoder.cnn_keys`.
- To log metrics from environments without showing them to the agent or storing
  them in the replay buffer, return them as observation keys with `log_` prefix
  and enable logging via the `run.log_keys_...` options.
- To continue stopped training runs, simply run the same command line again and
  make sure that the `--logdir` points to the same directory.

# Disclaimer

This repository contains a reimplementation of DreamerV3 based on the open
source DreamerV2 code base. It is unrelated to Google or DeepMind. The
implementation has been tested to reproduce the official results on a range of
environments.

[jax]: https://github.com/google/jax#pip-installation-gpu-cuda
[paper]: https://arxiv.org/pdf/2301.04104v1.pdf
[website]: https://danijar.com/dreamerv3
[tweet]: https://twitter.com/danijarh/status/1613161946223677441
[example]: https://github.com/danijar/dreamerv3/blob/main/example.py
