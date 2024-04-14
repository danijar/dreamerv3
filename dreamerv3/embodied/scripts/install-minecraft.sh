#!/bin/sh
set -eu

apt-get update
apt-get install -y libgl1-mesa-dev
apt-get install -y libx11-6
apt-get install -y openjdk-8-jdk
apt-get install -y x11-xserver-utils
apt-get install -y xvfb
apt-get clean

pip3 install minerl==0.4.4
