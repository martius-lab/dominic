#!/bin/bash

source /home/robot/.bashrc
conda activate isaac_gym_public
source /home/robot/ws_solo12/workspace/install/setup.bash
# change this on the hardware computer
cd /home/robot/Solo_workspace/Workspace/solo_legged_gym/legged_gym/scripts/solo12/
export DISPLAY=:1
python deployment.py solo12_config.yml