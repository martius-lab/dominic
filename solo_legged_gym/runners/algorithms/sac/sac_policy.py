import torch
import torch.nn as nn
from solo_legged_gym.runners.utils.distributions import SquashedDiagGaussianDistribution

LOG_STD_MAX = 0
LOG_STD_MIN = -20


class SACPolicy(nn.Module):
    #

    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation='elu',
                 **kwargs):
        if kwargs:
            print("SACPolicy.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(SACPolicy, self).__init__()

        activation = get_activation(activation)

        mlp_input_dim = num_obs

        # Policy
        layers = []
        layers.append(nn.Linear(mlp_input_dim, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
            layers.append(activation)
        self.policy_latent_net = nn.Sequential(*layers)

        self.distribution = SquashedDiagGaussianDistribution(action_dim=num_actions)
        self.action_mean_net = nn.Linear(hidden_dims[-1], num_actions)
        self.log_std_net = nn.Linear(hidden_dims[-1], num_actions)

        print(f"Policy Latent MLP: {self.policy_latent_net}\n")
        print(f"Policy Action Mean: {self.action_mean_net}\n")
        print(f"Policy Action log std: {self.log_std_net}\n")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_std(self):
        return self.distribution.distribution.stddev

    def get_action_dist_params(self, observations):
        policy_latent = self.policy_latent_net(observations)
        mean_actions = self.action_mean_net(policy_latent)
        log_std = torch.clamp(self.log_std_net(policy_latent), LOG_STD_MIN, LOG_STD_MAX)
        return mean_actions, log_std

    def act_and_log_prob(self, observations):
        # return actions and log_prob
        mean_actions, log_std = self.get_action_dist_params(observations)
        return self.distribution.log_prob_from_params(mean_actions=mean_actions, log_std=log_std)

    def act(self, observations):
        # return actions and log_prob
        mean_actions, log_std = self.get_action_dist_params(observations)
        return self.distribution.actions_from_params(mean_actions=mean_actions, log_std=log_std, deterministic=False)

    def act_inference(self, observations):
        mean_actions, log_std = self.get_action_dist_params(observations)
        return self.distribution.actions_from_params(mean_actions=mean_actions, log_std=log_std, deterministic=True)


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
