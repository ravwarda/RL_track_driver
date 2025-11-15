from PPO_network import PPO_Network
from ac_comunication import AC_Connection

import numpy as np
import torch
from torch.distributions import MultivariateNormal
from torch.nn import MSELoss


class PPO:
    def __init__(self, enviroment, input_size, output_size, device=None):
        # device selection
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Environment
        self.obs_dim = input_size
        self.action_dim = output_size
        self.env = enviroment

        # Initialize actor and critic networks
        self.actor = PPO_Network(self.obs_dim, self.action_dim).to(self.device)
        self.critic = PPO_Network(self.obs_dim, 1).to(self.device)

        # Hyperparameters
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1000
        self.gamma = 0.95
        self.clip = 0.2
        self.epochs_per_iteration = 5
        self.lr_actor = 0.0003
        self.lr_critic = 0.0003

        # ensure cov on correct device
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5, device=self.device)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic)

    def learn(self, total_steps):
        current_step = 0
        while current_step < total_steps:
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )

            current_step += np.sum(batch_lens)

            V, _ = self.evaluate(batch_obs, batch_acts)

            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.epochs_per_iteration):
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = (
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                )

                # Calculate actor loss
                actor_loss = -torch.min(surr1, surr2).mean()

                # Update actor network
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                # Calculate critic loss
                critic_loss = MSELoss()(V.squeeze(), batch_rtgs)

                # Update critic network
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

    def get_action(self, obs):
        mean = self.actor.forward(obs)

        dist = MultivariateNormal(mean, self.cov_mat)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        action = torch.tanh(action)

        # return numpy action and scalar log-prob (on CPU) for storage
        return action.detach().cpu().numpy(), float(log_prob.detach().cpu().item())

    def evaluate(self, batch_obs, batch_acts):
        # ensure inputs are on the correct device
        if isinstance(batch_obs, torch.Tensor):
            batch_obs = batch_obs.to(self.device)
        if isinstance(batch_acts, torch.Tensor):
            batch_acts = batch_acts.to(self.device)

        mean = self.actor.forward(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        v = self.critic.forward(batch_obs).squeeze()

        return v, log_probs

    def compute_rtgs(self, batch_rews):
        batch_rtgs = []

        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for reward in reversed(ep_rews):
                discounted_reward = reward + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float, device=self.device)
        return batch_rtgs

    def rollout(self):
        # Batch data
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        step = 0
        while step < self.timesteps_per_batch:

            ep_rews = []  # rewards for current episode

            obs = self.env.reset()

            for ep_step in range(self.max_timesteps_per_episode):
                step += 1

                batch_obs.append(obs)

                action, log_prob = self.get_action(obs)
                obs, reward = self.env.control_step(action)

                ep_rews.append(reward)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

            batch_lens.append(ep_step + 1)
            batch_rews.append(ep_rews)

        # move tensors to selected device
        batch_obs = torch.tensor(batch_obs, dtype=torch.float, device=self.device)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float, device=self.device)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float, device=self.device)

        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def save(self, path, include_optimizers = True):
        checkpoint = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "cov_var": self.cov_var.detach().cpu(),
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
        }
        if include_optimizers:
            checkpoint["actor_optimizer"] = self.actor_optimizer.state_dict()
            checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()

        torch.save(checkpoint, path)

    def load(self, path, load_optimizers = True):

        map_loc = self.device 
        checkpoint = torch.load(path, map_location=map_loc)

        # load network weights
        self.actor.load_state_dict(checkpoint["actor_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])

        # ensure models are on the correct device
        self.actor.to(self.device)
        self.critic.to(self.device)

        # load covariances
        if "cov_var" in checkpoint:
            self.cov_var = checkpoint["cov_var"].to(self.device)
            self.cov_mat = torch.diag(self.cov_var).to(self.device)

        # optionally load optimizers and move optimizer tensors to device
        if load_optimizers and "actor_optimizer" in checkpoint and "critic_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

            # move optimizer state tensors to the correct device
            def _move_opt_state(opt):
                for state in opt.state.values():
                    for k, v in list(state.items()):
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)

            _move_opt_state(self.actor_optimizer)
            _move_opt_state(self.critic_optimizer)