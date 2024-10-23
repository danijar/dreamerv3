#!/bin/sh
set -eu

apt-get update
apt-get install -y openjdk-8-jdk
apt-get clean

pip install https://github.com/danijar/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl

# apt-get update
# apt-get install -y openjdk-8-jdk
# apt-get clean
# pip install git+https://github.com/danijar/minerl.git@10f8d48
