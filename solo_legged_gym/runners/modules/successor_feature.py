import torch
import torch.nn as nn


class SuccessorFeature(nn.Module):
    def __init__(self,
                 num_obs,
                 num_features,
                 hidden_dims=[256, 256, 256],
                 activation='elu',
                 **kwargs):
        if kwargs:
            print("SuccessorFeature.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(SuccessorFeature, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs
        mlp_output_dim = num_features

        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], mlp_output_dim))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)
        self.succ_feat = nn.Sequential(*layers)

        print(f"Successor Feature MLP: {self.succ_feat}")

    @staticmethod
    def init_weights(sequential, scales):
        # not used at the moment
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        # not used at the moment
        pass

    def forward(self, x):
        return self.succ_feat(x)


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
