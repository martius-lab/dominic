## Installation
1. Clone the repo:
    ```bash
    git clone git@gitlab.is.tue.mpg.de:autonomous-learning/solo_legged_gym.git
    ```
    then checkout to this branch and go to the root folder.

2. Install poetry
    
    on local machine
    ```bash
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
    ```
    
    on cluster
    ```bash
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/fast/username/poetry python3 - --version 1.4.0
    ```
    Make sure to replace `username` by your account name. Installing in the `/fast/` directory gives you better execution speed.
    Newer versions of poetry might not work due to incompatbility with the specific package versions we require.

3. Set up bashrc

    on pc
    ```bash
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc 
    ```
    on cluster
    ```bash
    echo 'export PATH="/fast/<username>/poetry/bin:$PATH"' >> ~/.bashrc
    echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring' >> ~/.bashrc
    source ~/.bashrc 
    ```
    Make sure to replace the username in the bashrc with your own.

4. (Optional) Make sure you have setup ssh key access to `gitlab.tuebingen.mpg.de` and `gitlab.is.tue.mpg.de`. Both are needed for the required dependencies. If there are issues, try adding:
    ```
    eval $(ssh-agent -s)
    ssh-add ~/.ssh/mpi_is_gitlab
    ssh-add ~/.ssh/mpi_tuebingen_gitlab
    ```
    to you bashrc. Make sure to replace the key names by your own ssh-keys.

5. Check if this worked so far. Call
    ```bash
    poetry --version
    ```
    
    It should be:
    ```
    Poetry (version 1.4.0)
    ```

6. Now we want to use Python 3.8 to create virtual environment. This is ensured to work with isaacgym preview version 4. `python >= 3.10` will definitely not work with the cluster_utils package. First make sure that you are in a clean environment. Deactivate all conda env virtualenv environments and delete any `.venv` in the root folder of your legged_gym clone. Call:
    ```bash
    poetry env use /usr/bin/python3.8
    ```

7. Run 
    ```bash
    poetry install
    ```

8. Login Weights and Biases
   ```bash
   echo 'export WANDB_USERNAME=<wandb_username>' >> ~/.bashrc
   source ~/.bashrc 
   poetry run wandb login
   ```
   you will be asked to paste the API keys, you can get it from your personal profile. 

## Optional
1. Open the project with PyCharm and choose python interpreter as poetry. 
2. Configure PyCharm Run/Debug setting. 
   by setting the environment variables (important!)
   ```
   PYTHONUNBUFFERED=1;WANDB_USERNAME=<your-wandb-username>
   ```
3. Other troubleshooting
     To fix the following error:
     ```
     ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory
     ```
     install python-dev:
     ```bash
     sudo apt install libpython3.8
     ```

## RUN on local machine
Run the scripts from the root directory (where `.venv` is). 
1. use `scripts/train.py` to start training.
   For example:
   ```bash 
   poetry run python solo_legged_gym/scripts/train.py --task=solo12_vanilla (--wandb --headless)
   ```
   When viewer is enabled, use `v` to pause/resume rendering; use `b` to zoom in the first env/ zoom out. 
2. use `scripts/keyboard_play.py` to control the robot with trained model. 
   For example:
   ```bash 
   poetry run python solo_legged_gym/scripts/keyboard_play.py --task=solo12_vanilla
   ```
   use `w/a/s/d/q/e/x` to send command, specify the log data in `keyboard_play.py`
3. use `scripts/log_data_plot.py` to plot the data. 
   For example:
   ```bash
   poetry run python solo_legged_gym/scripts/log_data_plot.py --task=solo12_vanilla
   ```
   use `w/a/s/d/q/e/x` to send command, specify the log data in `keyboard_play.py`
4. If wandb is not used, tensorboard will be the default writer. 
   For example:
   ```bash
   poetry run tensorboard --logdir logs/solo12_vanilla
   ```
5. specify the argument as shown in `utis/helpers.py`
   
   ```python
   custom_parameters = [
           {
               "name": "--task",
               "type": str,
               "default": "solo12_vanilla",
               "help": "Start testing from a checkpoint. Overrides config file if provided.",
           },
           {
               "name": "--load_run",
               "type": str,
               "help": "Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
           },
           {
               "name": "--checkpoint",
               "type": int,
               "help": "Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
           },
           {
               "name": "--headless",
               "action": "store_true",
               "default": False,
               "help": "Force display off at all times",
           },
           {
               "name": "--device",
               "type": str,
               "default": "cuda:0",
               "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
           },
           {
               "name": "--wandb",
               "action": "store_true",
               "default": False,
               "help": "Turn on Weights and Biases writer. If disabled, tensorboard will be used. ",
           }
   ]
   ```

## Cluster run
run at least once on the local machine to update the json file specified in `env`

####  - Interactive debug on cluster
launch an interactive session to debug
```bash
condor_submit_bid 15 -i -append request_cpus=10 -append request_memory=20000 -append request_gpus=1
```
####  - Grid search
For training, specify the grid search params in `cluster/grid_search.json`
be sure to enable `tmux` so that your session will not terminate if you close the terminal.
```bash
poetry run python -m cluster.grid_search solo_legged_gym/cluster/grid_search.json
```
by default, weights and biases is used. 

#### - File transfer
FileZilla is recommended, you can easily install it
```bash
sudo apt install filezilla
```
connect the cluster by specifying `SFTP` and port `22`. 

## Deployment
#### Solo 12 System

The Solo 12 operating system runs on `Olympus`.

```
Username: robot
Password: S0loeight
```

#### Hardware

The hardware details are documented [here](https://github.com/open-dynamic-robot-initiative/open_robot_actuator_hardware).

#### Software

Set up `solo_legged_gym` environment in `~/solo12_workspace`.

The low-level robot interface has been compiled and is ready to be used. To recompile the workspace, go to `~/ws_solo12/workspace` and do `colcon build`. Remember to import the ROS installation in by `source /opt/ros/foxy/setup.bash`.

#### Operational Instruction

1. Set up the [Vicon System](https://gitlab.is.tue.mpg.de/autonomous-learning/wiki/-/wikis/Vicon).

2. Put the robot on the stand holder. Turn on the robot power supply and release the emergency stop.

3. Set up robot configurations in `solo12_config.yml`.

   > The workstation communicates with Solo 12 using an Ethernet cable. Set `network_interface` to the correct interface.

   > To calibrate joint positions, measure `home_offset_rad` by running `ros2 run solo solo12_hardware_calibration network_interface` in `ws_solo12`.

4. If you have already played the learned policy with `script/keyboard_play.py`, you should have the exported policy in the log folder where there should be a folder `exported`.
   
   > Change the load policy in `deployment/play.py`. 

5. Adapt observation space in `_compute_observations()` and key commands in `_on_press()` in `deployment/play.py`.

   > Please go through `deployment.py` to check other configurations (default joint positions, PD gains, etc.) and adapt where necessary.

6. Obtain root access by executing `sudo -i`.

7. Move the robot legs close to the zero position. Executing the script in `run_robot.bash` in the root panel. Use the keyboard to update commands.

   > Unlike Solo 8, the homing of the joints for Solo 12 is integrated and is executed at the beginning of each run. The joint offset values are remembered each time when powered on. Therefore, to redo homing, the robot should be rebooted entirely.

```
#!/bin/bash

source /home/robot/.bashrc
source /home/robot/ws_solo12/workspace/install/setup.bash
source /home/robot/solo12_workspace/solo_legged_gym/.venv/bin/activate
cd /home/robot/solo12_workspace/solo_legged_gym/solo_legged_gym/deployment
export DISPLAY=:1
python play.py solo12_config.yml
```

8. Switch off the power supply, press the emergency stop, and switch off the Vicon system after experiments.

## WIP
1. Cluster optimization
2. how to visualize the cluster training using tensorboard? (to limit the internet usage)

