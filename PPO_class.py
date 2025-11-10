from PPO_network import PPO_Network

import torch
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self, input_size, output_size):
        # Environment
        self.obs_dim = input_size
        self.action_dim = output_size

        # Initialize actor and critic networks
        self.actor = PPO_Network(input_size, output_size)
        self.critic = PPO_Network(input_size, 1)

        # Hyperparameters
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1000

        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def learn(self, total_steps):
        current_step = 0
        while current_step < total_steps:
            pass

    def get_action(self, obs):
        mean = self.actor.forward(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()
        
    def rollout(self):
        # Batch data
        batch_obs = []             # batch observations
        batch_acts = []            # batch actions
        batch_log_probs = []       # log probs of each action
        batch_rews = []            # batch rewards
        batch_rtgs = []            # batch rewards-to-go
        batch_lens = []            # episodic lengths in batch

        step = 0
        while step < self.timesteps_per_batch:

            ep_rews = []          # rewards for current episode

            done = False
            obs = None  # Placeholder for environment reset

            for ep_step in range(self.max_timesteps_per_episode):
                step += 1
                
                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward, done = None, None, None  # Placeholder for environment interaction

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break

            batch_lens.append(ep_step + 1)
            batch_rews.append(ep_rews)


