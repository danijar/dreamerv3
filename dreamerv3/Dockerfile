# Prerequisites: Nsuyre you have installed NVIDIA Container Toolkit as per https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
#
# 1. Test setup:
# docker run -it --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi
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
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1
RUN apt-get update && apt-get install -y \
  ffmpeg git python3-pip vim libglew-dev \
  x11-xserver-utils xvfb curl libegl1-mesa \
  && apt-get clean

# TensorFlow Install 
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
ENV PATH /opt/conda/bin:$PATH
RUN conda update -n base -c defaults conda
RUN conda install -c conda-forge cudatoolkit=11.8.0
RUN python -m pip install nvidia-cudnn-cu11==8.6.0.163 tensorflow==2.13.*
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
RUN echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
RUN bash -c "source $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh"

RUN pip3 install --upgrade pip

# Envs
ENV MUJOCO_GL=osmesa
COPY dreamerv3/embodied/scripts scripts
RUN sh scripts/install-dmlab.sh
# RUN sh scripts/install-atari.sh
# RUN sh scripts/install-minecraft.sh
ENV NUMBA_CACHE_DIR=/tmp
RUN pip3 install crafter
RUN pip3 install dm_control
RUN pip3 install robodesk
RUN pip3 install bsuite

# Agent
RUN pip3 install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip3 install jaxlib
RUN pip3 install tensorflow_probability
RUN pip3 install optax
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
