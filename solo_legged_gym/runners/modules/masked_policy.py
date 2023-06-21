import torch
import torch.nn as nn
from solo_legged_gym.runners.utils.distributions import DiagGaussianDistribution


class MaskedPolicy(nn.Module):
    def __init__(self,
                 num_obs,
                 num_skills,
                 num_actions,
                 share_ratio,
                 hidden_dims=None,
                 activation='elu',
                 log_std_init=0.0,
                 device='cpu',
                 **kwargs):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if kwargs:
            print("MaskedPolicy.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(MaskedPolicy, self).__init__()

        activation = get_activation(activation)
        mlp_input_dim = num_obs

        self.num_hidden_dim = len(hidden_dims)
        self.device = device

        # Policy
        self.policy_latent_layers = nn.ModuleList()
        self.policy_latent_layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]).to(self.device))
        self.policy_latent_layers.append(activation)
        for la in range(self.num_hidden_dim - 1):
            self.policy_latent_layers.append(nn.Linear(hidden_dims[la], hidden_dims[la + 1]).to(self.device))
            self.policy_latent_layers.append(activation)

        self.distribution = DiagGaussianDistribution(action_dim=num_actions)
        self.action_mean_net, self.log_std = self.distribution.proba_distribution_net(latent_dim=hidden_dims[-1],
                                                                                      log_std_init=log_std_init)

        # Mask
        self.masks = nn.ParameterList()
        for la in range(self.num_hidden_dim):
            self.masks.append(torch.nn.Parameter((torch.rand((num_skills, hidden_dims[la])) < share_ratio).float(),
                                                 requires_grad=False).to(self.device))

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.distribution.mean

    @property
    def action_std(self):
        return self.distribution.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy()

    def act_and_log_prob(self, input_x):
        # return actions and log_prob
        x, z = input_x
        skill_idxs = z.argmax(dim=1).flatten()
        batched_masks = [self.masks[la][skill_idxs] for la in range(self.num_hidden_dim)]
        for la in range(self.num_hidden_dim):
            x = self.policy_latent_layers[2*la+1](self.policy_latent_layers[2*la](x)) * batched_masks[la]
        mean = self.action_mean_net(x)
        return self.distribution.log_prob_from_params(mean_actions=mean, log_std=self.log_std)

    def act_inference(self, input_x):
        x, z = input_x
        skill_idxs = z.argmax(dim=1).flatten()
        batched_masks = [self.masks[la][skill_idxs] for la in range(self.num_hidden_dim)]
        for la in range(self.num_hidden_dim):
            x = self.policy_latent_layers[2*la+1](self.policy_latent_layers[2*la](x)) * batched_masks[la]
        mean = self.action_mean_net(x)
        return self.distribution.actions_from_params(mean_actions=mean, log_std=self.log_std, deterministic=True)


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