#!/bin/bash

source /home/robot/.bashrc
source /home/robot/ws_solo12/workspace/install/setup.bash
source /home/robot/solo12_workspace/solo_legged_gym/.venv/bin/activate
cd /home/robot/solo12_workspace/solo_legged_gym/solo_legged_gym/deployment
export DISPLAY=:1
python play.py solo12_config.yml