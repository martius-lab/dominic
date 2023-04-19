import torch
import torch.nn as nn
from solo_legged_gym.runners.utils.distributions import DiagGaussianDistribution


class PPOPolicy(nn.Module):
    def __init__(self,
                 num_obs,
                 num_actions,
                 hidden_dims=[256, 256, 256],
                 activation='elu',
                 log_std_init=0.0,
                 **kwargs):
        if kwargs:
            print("PPOPolicy.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        super(PPOPolicy, self).__init__()

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

        self.distribution = DiagGaussianDistribution(action_dim=num_actions)
        self.action_mean_net, self.log_std = self.distribution.proba_distribution_net(latent_dim=hidden_dims[-1], log_std_init=log_std_init)

        print(f"Policy Latent MLP: {self.policy_latent_net}\n")
        print(f"Policy Action Mean: {self.action_mean_net}\n")
        print(f"Policy Action log std: {self.log_std}\n")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mode()

    @property
    def action_std(self):
        return torch.ones_like(self.action_mean) * self.log_std.exp()

    @property
    def entropy(self):
        return self.distribution.entropy()

    def act(self, observations):
        # return actions and log_prob
        mean = self.action_mean_net(self.policy_latent_net(observations))
        return self.distribution.log_prob_from_params(mean_actions=mean, log_std=self.log_std)

    def act_inference(self, observations):
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
