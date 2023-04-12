### Installation
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
    export PATH="/fast/<username>/poetry/bin:$PATH"
    export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
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

### Optional
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

### RUN on local machine
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
               "default": "a1_vanilla",
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

### Cluster run
run at least once on the local machine to update the json file specified in `env`

####  - Interactive debug on cluster
launch an interactive session to debug
```bash
condor_submit_bid 15 -i -append request_cpus=10 -append request_memory=20000 -append request_gpus=1
export PATH="/fast/username/poetry/bin:$PATH"
```
####  - Grid search
For training, specify the grid search params in `cluster/grid_search.json`
be sure to enable `tmux` so that your session will not terminate if you close the terminal.
```bash
export PATH="/fast/username/poetry/bin:$PATH"
poetry run python -m cluster.grid_search solo_legged_gym/cluster/grid_search.json
```
by default, weights and biases is used. 

### WIP
1. Cluster optimization
2. how to visualize the cluster training using tensorboard? (to limit the internet usage)

