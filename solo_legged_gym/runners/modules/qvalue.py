import torch
import torch.nn as nn


class QValue(nn.Module):
    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation='elu',
                 **kwargs):
        if kwargs:
            print("QValue.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(QValue, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs + num_actions

        # Value function
        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], 1))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.qvalue = nn.Sequential(*layers)

        print(f"Q-Value MLP: {self.qvalue}")

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        # not used at the moment
        pass

    def forward(self):
        # not used at the moment
        raise NotImplementedError

    def evaluate(self, observations, actions):
        qvalue = self.qvalue(torch.cat([observations, actions], dim=-1))
        return qvalue


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
