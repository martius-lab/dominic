import torch


class ReplayBuffer:
    class Transition:
        def __init__(self):
            self.observations = None
            self.next_observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.timeouts = None

        def clear(self):
            self.__init__()

    def __init__(self, num_envs,
                 num_transitions_per_env,
                 obs_shape,
                 actions_shape,
                 device='cpu'):

        # buffer size = num_transitions_per_env * num_envs
        self.device = device
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.actions_shape = actions_shape

        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.next_observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()
        self.timeouts = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device).byte()

        self.step = 0
        self.full = False

    def add_transitions(self, transition: Transition):
        self.observations[self.step].copy_(transition.observations)
        self.next_observations[self.step].copy_(transition.next_observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(-1, 1))
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.timeouts[self.step].copy_(transition.timeouts.view(-1, 1))

        self.step += 1
        if self.step >= self.num_transitions_per_env:
            self.full = True
            self.step = 0

    # def clear(self):
    #     self.step = 0

    def mini_batch_generator(self, mini_batch_size, num_mini_batches, num_epochs=8):
        buffer_size = self.num_envs * self.num_transitions_per_env

        if self.full:
            indices = (torch.randint(low=1, high=buffer_size, size=[num_mini_batches * mini_batch_size],
                                     requires_grad=False, device=self.device) + self.step) % buffer_size
        else:
            indices = torch.randint(low=0, high=self.step, size=[num_mini_batches * mini_batch_size],
                                    requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        next_observations = self.next_observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        rewards = self.rewards.flatten(0, 1)
        dones = self.dones.flatten(0, 1)
        timeouts = self.timeouts.flatten(0, 1)

        for epoch in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                obs_batch = observations[batch_idx]
                next_obs_batch = next_observations[batch_idx]
                actions_batch = actions[batch_idx]
                rewards_batch = rewards[batch_idx]
                # Only use dones that are not due to timeouts
                dones_batch = dones[batch_idx] * (1 - timeouts[batch_idx])

                yield obs_batch, next_obs_batch, actions_batch, rewards_batch, dones_batch
