import torch
import torch.nn as nn


class MaskedSuccessorFeature(nn.Module):
    def __init__(self,
                 num_obs,
                 num_skills,
                 num_features,
                 share_ratio,
                 hidden_dims=None,
                 activation='elu',
                 device='cpu',
                 **kwargs):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if kwargs:
            print("MaskedSuccessorFeature.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(MaskedSuccessorFeature, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim = num_obs
        mlp_output_dim = num_features

        self.num_hidden_dim = len(hidden_dims)
        self.device = device

        # MLP
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]).to(self.device))
        self.layers.append(activation)
        for la in range(self.num_hidden_dim):
            if la == self.num_hidden_dim - 1:
                self.layers.append(nn.Linear(hidden_dims[la], mlp_output_dim).to(self.device))
            else:
                self.layers.append(nn.Linear(hidden_dims[la], hidden_dims[la + 1]).to(self.device))
                self.layers.append(activation)

        # Mask
        self.masks = nn.ParameterList()
        for la in range(self.num_hidden_dim):
            self.masks.append(torch.nn.Parameter((torch.rand((num_skills, hidden_dims[la])) <= share_ratio).float(),
                                                 requires_grad=False).to(self.device))

    @staticmethod
    def init_weights(sequential, scales):
        # not used at the moment
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        # not used at the moment
        pass

    def forward(self, input_x):
        x, z = input_x
        skill_idxs = z.argmax(dim=1).flatten()
        batched_masks = [self.masks[la][skill_idxs] for la in range(self.num_hidden_dim)]
        for la in range(self.num_hidden_dim):
            x = self.layers[2 * la + 1](self.layers[2 * la](x)) * batched_masks[la]
        return self.layers[-1](x)


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
