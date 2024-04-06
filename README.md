# Learning Diverse Skills for Local Navigation under Multi-constraint Optimality (ICRA-2024)

**_[Website](https://sites.google.com/view/icra2024-dominic)_**  | **_[Paper](https://arxiv.org/abs/2310.02440)_**  | **_[Supplementary Video](https://youtu.be/DXNqdx4SsdA?si=ux_jLKuoodmp_fxW)_**

Authors:
[Jin Cheng](https://jin-cheng.me/), [Marin Vlastelica](https://jimimvp.github.io/), [Pavel Kolev](https://pavelkolev.github.io/), [Chenhao Li](https://breadli428.github.io/), [Georg Martius](https://is.mpg.de/person/gmartius)

Autonomous Learning Group, Max Planck Institute for Intelligent Systems, Tübingen, Germany

ETH Zürich, Zürich, Switzerland

University of Tübingen, Tübingen, Germany

## Installation
1. Clone the repo:
    ```bash
    git clone git@gitlab.is.tue.mpg.de:autonomous-learning/solo_legged_gym.git
    ```

2. Install poetry
    
    ```bash
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    source ~/.bashrc 
    ```

3. Check if this worked so far. Call
    ```bash
    poetry --version
    ```
    
    It should be:
    ```
    Poetry (version 1.4.0)
    ```

4. Now we want to use Python 3.8 to create virtual environment. This is ensured to work with isaacgym preview version 4. First make sure that you are in a clean environment. Deactivate all conda env virtualenv environments and delete any `.venv` in the root folder of your legged_gym clone. Call:
    ```bash
    poetry env use /usr/bin/python3.8
    ```
   
   If get stuck in creating venv in endless loop, try
   ```bash
   export XDG_DATA_HOME=/tmp
   ```
5. Install dependencies 
    ```bash
    poetry install
    ```
6. Download IsaacGym from [here](https://developer.nvidia.com/isaac-gym/download) and extract it to somewhere you like :). 
   ```bash
   poetry shell  # enter the virtual environment
   ```
   enter the directory of `IsaacGym_Preview_4_Package/isaacgym/python`, and run
   ```bash
   pip install -e .
   ```
   Don't panic if some packages are removed and reinstalled :)

7. (Optional) Login Weights and Biases
   ```bash
   echo 'export WANDB_USERNAME=<wandb_username>' >> ~/.bashrc
   source ~/.bashrc 
   poetry run wandb login
   ```
   you will be asked to paste the API keys, you can get it from your personal profile. 

## Troubleshooting
   To fix the following error:
   ```
   ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory
   ```
   install python-dev:
   ```bash
   sudo apt install libpython3.8
   ```

## Run the code
Run the scripts from the repository root directory (where `.venv` is). 

You can either choose to enter the environment by runing `poetry shell` or run the scripts directly by `poetry run python <script_name>`.

1. use `scripts/train.py` to start training.
   For example:
   ```bash 
   python solo_legged_gym/scripts/train.py --task=solo12_dominic_position --w
   ```
   When viewer is enabled (which should be the case by default), use `v` to pause/resume rendering; use `b` to zoom in the first env/ zoom out. 

   In principle, you can specify `--w` to enable Weights&Biases and specify `--dv` to disable viewer.
   
   **Special Note on video recording functionality with Wandb:**
    - Using video recording may slow down the training process, but for visualization purpose, it is recommended to enable it.
    - To make sure video recording works properly, please enable wandb by '--w' and **DON'T** use `--dv` to disable viewer.
2. use `scripts/position_play.py` to play the trained skill. 
   For example:
   ```bash 
   python solo_legged_gym/scripts/position_play.py --task=solo12_dominic_position
   ```
   specify the log data in `position_play.py`, use `r` to restart, use number to specify skill.
3. use `scripts/position_play2.py` to play all trained skills. 
   For example:
   ```bash 
   python solo_legged_gym/scripts/position_play2.py --task=solo12_dominic_position
   ```
   specify the log data in `position_play2.py`, use `r` to restart, use number to specify skill.
4. If wandb is not used, tensorboard will be the default writer. 
   For example:
   ```bash
   run tensorboard --logdir logs/solo12_dominic_position
   ```

## Addtionally
This project is based on [legged_gym](https://github.com/leggedrobotics/legged_gym) and [rsl_rl](https://github.com/leggedrobotics/rsl_rl). We thank all the contributors for their great work.