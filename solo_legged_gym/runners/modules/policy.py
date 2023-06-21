import torch
import torch.nn as nn
from solo_legged_gym.runners.utils.distributions import DiagGaussianDistribution


class Policy(nn.Module):
    def __init__(self,
                 num_obs,
                 num_skills,
                 num_actions,
                 hidden_dims=None,
                 activation='elu',
                 log_std_init=0.0,
                 device='cpu',
                 **kwargs):
        if hidden_dims is None:
            hidden_dims = [256, 256]
        if kwargs:
            print("Policy.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(Policy, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs + num_skills

        # Policy
        layers = [nn.Linear(mlp_input_dim, hidden_dims[0]).to(device), activation]
        for la in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[la], hidden_dims[la + 1]).to(device))
            layers.append(activation)
        self.policy_latent_net = nn.Sequential(*layers)

        self.distribution = DiagGaussianDistribution(action_dim=num_actions)
        self.action_mean_net, self.log_std = self.distribution.proba_distribution_net(latent_dim=hidden_dims[-1],
                                                                                      log_std_init=log_std_init)

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
        observations = torch.concat((x, z), dim=-1)
        mean = self.action_mean_net(self.policy_latent_net(observations))
        return self.distribution.log_prob_from_params(mean_actions=mean, log_std=self.log_std)

    def act_inference(self, input_x):
        x, z = input_x
        observations = torch.concat((x, z), dim=-1)
        mean = self.action_mean_net(self.policy_latent_net(observations))
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
