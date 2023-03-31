### Installation
0. Clone the repo:
    ```bash
    git clone git@gitlab.is.tue.mpg.de:autonomous-learning/solo_legged_gym.git
    ```
    then go to the root folder.

1. Install poetry
    
    on PC
    ```bash
    curl -sSL https://install.python-poetry.org | python3 - --version 1.4.0
    ```
    
    on cluster
    ```bash
    curl -sSL https://install.python-poetry.org | POETRY_HOME=/fast/username/poetry python3 - --version 1.4.0
    ```
    Make sure to replace `username` by your account name. Installing in the `/fast/` directory gives you better execution speed.
    Newer versions of poetry might not work due to incompatbility with the specific package versions we require.

2. Set up bashrc

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

3. (Optional) Make sure you have setup ssh key access to `gitlab.tuebingen.mpg.de` and `gitlab.is.tue.mpg.de`. Both are needed for the required dependencies. If there are issues, try adding:
    ```
    eval $(ssh-agent -s)
    ssh-add ~/.ssh/mpi_is_gitlab
    ssh-add ~/.ssh/mpi_tuebingen_gitlab
    ```
    to you bashrc. Make sure to replace the key names by your own ssh-keys.

4. Check if this worked so far. Call
    ```bash
    poetry --version
    ```
    
    It should be:
    ```
    Poetry (version 1.4.0)
    ```

5. Now we want to use Python 3.8 to create virtual environment. This is ensured to work with isaacgym preview version 4. `python >= 3.10` will definitely not work with the cluster_utils package. First make sure that you are in a clean environment. Deactivate all conda env virtualenv environments and delete any `.venv` in the root folder of your legged_gym clone. Call:
    ```bash
    poetry env use /usr/bin/python3.8
    ```

6. Run 
    ```bash
    poetry install
    ```

6. Login Weights and Biases
   ```bash
   wandb login
   ```
   
7. (optional) Configure pycharm by setting the environment variables (important)
   ```
   PYTHONUNBUFFERED=1;WANDB_USERNAME=<your-wandb-username>
   ```

8. Other troubleshooting
    To fix the following error:
    ```
    ImportError: libpython3.8m.so.1.0: cannot open shared object file: No such file or directory
    ```
    install python-dev:
    ```bash
    sudo apt install libpython3.8
    ```

### RUN
use `script/train.py` to start training.

use `script/play.py` to test a trained model.

use `script/keyboard_play.py` to control the robot with trained model. 

specify the argument as shown in `utis/helpers.py`

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
            "help": "Turn on Weights and Bias writer",
        }
]
```


### WIP:

1. cluster integration with json files
2. peotry integration
