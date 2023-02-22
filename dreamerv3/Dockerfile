# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.4.2-cudnn8-runtime-ubuntu20.04 nvidia-smi
#
# If the above does not work, try adding the --privileged flag
# and changing the command to `sh -c 'ldconfig -v && nvidia-smi'`.
#
# 2. Start training:
# docker build -f  dreamerv3/Dockerfile -t img . && \
# docker run -it --rm --gpus all -v ~/logdir:/logdir img \
#   sh scripts/xvfb_run.sh python3 dreamerv3/train.py \
#   --logdir "/logdir/$(date +%Y%m%d-%H%M%S)" \
#   --configs dmc_vision --task dmc_walker_walk
#
# 3. See results:
# tensorboard --logdir ~/logdir

# System
FROM nvidia/cuda:11.4.2-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb \
  && apt-get clean
RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL egl
ENV DMLAB_DATASET_PATH /dmlab_data
COPY scripts scripts
RUN sh scripts/install-dmlab.sh
RUN sh scripts/install-atari.sh
RUN sh scripts/install-minecraft.sh
ENV NUMBA_CACHE_DIR=/tmp
RUN pip3 install crafter
RUN pip3 install dm_control
RUN pip3 install robodesk
RUN pip3 install bsuite

# Agent
RUN pip3 install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jaxlib
RUN pip3 install tensorflow_probability
RUN pip3 install optax
RUN pip3 install tensorflow-cpu
ENV XLA_PYTHON_CLIENT_MEM_FRACTION 0.8

# Google Cloud DNS cache (optional)
ENV GCS_RESOLVE_REFRESH_SECS=60
ENV GCS_REQUEST_CONNECTION_TIMEOUT_SECS=300
ENV GCS_METADATA_REQUEST_TIMEOUT_SECS=300
ENV GCS_READ_REQUEST_TIMEOUT_SECS=300
ENV GCS_WRITE_REQUEST_TIMEOUT_SECS=600

# Embodied
RUN pip3 install numpy cloudpickle ruamel.yaml rich zmq msgpack
COPY . /embodied
RUN chown -R 1000:root /embodied && chmod -R 775 /embodied

WORKDIR embodied
