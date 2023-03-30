### Install

1. create a python virtual env with python 3.8
2. install pytorch 
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
   ```

3. install Isaac Gym
- Download from https://developer.nvidia.com/isaac-gym
  ```bash
  cd isaacgym/python
  pip install -e .
  ```
  
- Try running an example:
  ```bash
  cd examples
  python 1080_balls_of_solitude.py
  ```
  
- For troubleshooting, check docs at: `isaacgym/docs/index.html`

4. install other libs
    ```bash
   pip install tensorboard
   pip install numpy==1.23
   pip install matplotlib
    ```
   
5. install solo_legged_gym as a package
    ```bash
   cd solo_legged_gym
   pip install -e .
    ```

6. install and login weights and bias
   ```bash
   pip install wandb
   wandb login
   ```
   
7. configure pycharm by setting the environment variables (important)
   ```
   PYTHONUNBUFFERED=1;LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path-to-virtual-env>/lib;WANDB_USERNAME=<your-wandb-username>
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


### TODO:

1. cluster integration with json files
2. peotry integration
