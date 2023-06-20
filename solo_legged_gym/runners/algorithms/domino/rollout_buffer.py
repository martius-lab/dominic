import torch


class RolloutBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.ext_rews = None  # list of extrinsic rewards
            self.ext_values = None  # list of extrinsic values
            self.int_rew = None
            self.int_values = None
            self.skills = None
            self.features = None
            self.succ_feat = None
            self.dones = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 actions_shape,
                 features_shape,
                 num_ext_values=1,
                 device='cpu'):

        self.device = device

        self.obs_shape = obs_shape
        self.actions_shape = actions_shape
        self.num_ext_values = num_ext_values

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)

        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

        self.int_rew = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.int_values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.int_returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.int_advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)

        self.ext_rews = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                         for _ in range(self.num_ext_values)]
        self.ext_values = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                           for _ in range(self.num_ext_values)]
        self.ext_returns = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                            for _ in range(self.num_ext_values)]
        self.ext_advantages = [torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
                               for _ in range(self.num_ext_values)]

        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.skills = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.long, device=self.device)
        self.features = torch.zeros(num_transitions_per_env, num_envs, *features_shape, device=self.device)
        self.succ_feat = torch.zeros(num_transitions_per_env, num_envs, *features_shape, device=self.device)
        self.succ_feat_target = torch.zeros(num_transitions_per_env, num_envs, *features_shape, device=self.device)

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
        self.int_values[self.step].copy_(transition.int_values)

        for i in range(self.num_ext_values):
            self.ext_rews[i][self.step].copy_(transition.ext_rews[i].view(-1, 1))
            self.ext_values[i][self.step].copy_(transition.ext_values[i])

        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.skills[self.step].copy_(transition.skills.view(-1, 1))
        self.features[self.step].copy_(transition.features)
        self.succ_feat[self.step].copy_(transition.succ_feat)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(self, last_ext_values, last_int_values, gamma, lam):
        ext_advantages = [0 for _ in range(self.num_ext_values)]
        int_advantage = 0

        for step in reversed(range(self.num_transitions_per_env)):

            if step == self.num_transitions_per_env - 1:
                next_ext_values = [last_ext_values[i] for i in range(self.num_ext_values)]
                next_int_values = last_int_values
            else:
                next_ext_values = [self.ext_values[i][step + 1] for i in range(self.num_ext_values)]
                next_int_values = self.int_values[step + 1]

            next_is_not_terminal = 1.0 - self.dones[step].float()

            ext_deltas = [0 for _ in range(self.num_ext_values)]
            for i in range(self.num_ext_values):
                ext_deltas[i] = self.ext_rews[i][step] + next_is_not_terminal * gamma * next_ext_values[i] - self.ext_values[i][step]
                ext_advantages[i] = ext_deltas[i] + next_is_not_terminal * gamma * lam * ext_advantages[i]
                self.ext_returns[i][step] = ext_advantages[i] + self.ext_values[i][step]

            int_delta = self.int_rew[step] + next_is_not_terminal * gamma * next_int_values - self.int_values[step]
            int_advantage = int_delta + next_is_not_terminal * gamma * lam * int_advantage
            self.int_returns[step] = int_advantage + self.int_values[step]

        # Compute and normalize the advantages
        for i in range(self.num_ext_values):
            self.ext_advantages[i] = self.ext_returns[i] - self.ext_values[i]
            self.ext_advantages[i] = (self.ext_advantages[i] - self.ext_advantages[i].mean()) / (self.ext_advantages[i].std() + 1e-8)
        self.int_advantages = self.int_returns - self.int_values
        self.int_advantages = (self.int_advantages - self.int_advantages.mean()) / (self.int_advantages.std() + 1e-8)

    def compute_succ_feat_target(self, last_succ_feat, gamma):
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_succ_feat = last_succ_feat
            else:
                next_succ_feat = self.succ_feat[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            self.succ_feat_target[step] = self.features[step] + next_is_not_terminal * gamma * next_succ_feat

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)

        ext_values = [self.ext_values[i].flatten(0, 1) for i in range(self.num_ext_values)]
        ext_returns = [self.ext_returns[i].flatten(0, 1) for i in range(self.num_ext_values)]
        ext_advantages = [self.ext_advantages[i].flatten(0, 1) for i in range(self.num_ext_values)]

        int_values = self.int_values.flatten(0, 1)
        int_returns = self.int_returns.flatten(0, 1)
        int_advantages = self.int_advantages.flatten(0, 1)

        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)
        skills = self.skills.flatten(0, 1)
        succ_feat_target = self.succ_feat_target.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                actions_batch = actions[batch_idx]
                old_actions_log_prob_batch = old_actions_log_prob[batch_idx]
                old_mu_batch = old_mu[batch_idx]
                old_sigma_batch = old_sigma[batch_idx]

                target_ext_values_batch = [ext_values[i][batch_idx] for i in range(self.num_ext_values)]
                ext_returns_batch = [ext_returns[i][batch_idx] for i in range(self.num_ext_values)]
                ext_advantages_batch = [ext_advantages[i][batch_idx] for i in range(self.num_ext_values)]

                target_int_values_batch = int_values[batch_idx]
                int_returns_batch = int_returns[batch_idx]
                int_advantages_batch = int_advantages[batch_idx]

                skills_batch = skills[batch_idx]
                succ_feat_target_batch = succ_feat_target[batch_idx]
                yield obs_batch, actions_batch, target_ext_values_batch, target_int_values_batch, \
                    ext_advantages_batch, int_advantages_batch, \
                    ext_returns_batch, int_returns_batch, \
                    old_actions_log_prob_batch, old_mu_batch, old_sigma_batch, skills_batch, succ_feat_target_batch
