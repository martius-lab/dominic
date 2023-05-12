import torch


class RolloutBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.fixed_rew = None
            self.loose_rew = None
            self.int_rew = None
            self.fixed_values = None
            self.loose_values = None
            self.int_values = None
            self.skills = None
            self.features = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 actions_shape,
                 features_shape,
                 device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.int_rew = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.fixed_rew = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.loose_rew = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.int_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.fixed_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.loose_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.int_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.fixed_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.loose_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.int_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.fixed_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.loose_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.skills = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.long, device=self.device)
        self.features = torch.zeros(num_transitions_per_env, num_envs, *features_shape, device=self.device)

        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.int_rew[self.step].copy_(transition.int_rew.view(-1, 1))
        self.fixed_rew[self.step].copy_(transition.fixed_rew.view(-1, 1))
        self.loose_rew[self.step].copy_(transition.loose_rew.view(-1, 1))
        self.int_values[self.step].copy_(transition.int_values)
        self.fixed_values[self.step].copy_(transition.fixed_values)
        self.loose_values[self.step].copy_(transition.loose_values)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.skills[self.step].copy_(transition.skills.view(-1, 1))
        self.features[self.step].copy_(transition.features)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_int_values, last_fixed_values, last_loose_values, gamma, lam):
        int_advantage = 0
        fixed_advantage = 0
        loose_advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_int_values = last_int_values
                next_fixed_values = last_fixed_values
                next_loose_values = last_loose_values
            else:
                next_int_values = self.int_values[step + 1]
                next_fixed_values = self.fixed_values[step + 1]
                next_loose_values = self.loose_values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            int_delta = self.int_rew[step] + next_is_not_terminal * gamma * next_int_values - self.int_values[step]
            fixed_delta = self.fixed_rew[step] + next_is_not_terminal * gamma * next_fixed_values - self.fixed_values[step]
            loose_delta = self.loose_rew[step] + next_is_not_terminal * gamma * next_loose_values - self.loose_values[step]
            int_advantage = int_delta + next_is_not_terminal * gamma * lam * int_advantage
            fixed_advantage = fixed_delta + next_is_not_terminal * gamma * lam * fixed_advantage
            loose_advantage = loose_delta + next_is_not_terminal * gamma * lam * loose_advantage
            self.int_returns[step] = int_advantage + self.int_values[step]
            self.fixed_returns[step] = fixed_advantage + self.fixed_values[step]
            self.loose_returns[step] = loose_advantage + self.loose_values[step]

        # Compute and normalize the advantages
        self.int_advantages = self.int_returns - self.int_values
        self.fixed_advantages = self.fixed_returns - self.fixed_values
        self.loose_advantages = self.loose_returns - self.loose_values
        self.int_advantages = (self.int_advantages - self.int_advantages.mean()) / (self.int_advantages.std() + 1e-8)
        self.fixed_advantages = (self.fixed_advantages - self.fixed_advantages.mean()) / (self.fixed_advantages.std() + 1e-8)
        self.loose_advantages = (self.loose_advantages - self.loose_advantages.mean()) / (self.loose_advantages.std() + 1e-8)

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        int_values = self.int_values.flatten(0, 1)
        fixed_values = self.fixed_values.flatten(0, 1)
        loose_values = self.loose_values.flatten(0, 1)
        int_returns = self.int_returns.flatten(0, 1)
        fixed_returns = self.fixed_returns.flatten(0, 1)
        loose_returns = self.loose_returns.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        int_advantages = self.int_advantages.flatten(0, 1)
        fixed_advantages = self.fixed_advantages.flatten(0, 1)
        loose_advantages = self.loose_advantages.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        skills = self.skills.flatten(0, 1)
        features = self.features.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                target_int_values_batch = int_values[batch_idx]
                target_fixed_values_batch = fixed_values[batch_idx]
                target_loose_values_batch = loose_values[batch_idx]
                int_returns_batch = int_returns[batch_idx]
                fixed_returns_batch = fixed_returns[batch_idx]
                loose_returns_batch = loose_returns[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                int_advantages_batch = int_advantages[batch_idx]
                fixed_advantages_batch = fixed_advantages[batch_idx]
                loose_advantages_batch = loose_advantages[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]
                skills_batch = skills[batch_idx]
                features_batch = features[batch_idx]
                yield obs_batch, actions_batch, \
                    target_int_values_batch, target_fixed_values_batch, target_loose_values_batch, \
                    int_advantages_batch, fixed_advantages_batch, loose_advantages_batch, \
                    int_returns_batch, fixed_returns_batch, loose_returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, skills_batch, features_batch
