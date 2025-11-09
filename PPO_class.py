from PPO_network import PPO_Network

class PPO:
    def __init__(self, input_size, output_size):
        # Initialize actor and critic networks
        self.actor = PPO_Network(input_size, output_size)
        self.critic = PPO_Network(input_size, 1)

        # Hyperparameters
        self.timesteps_per_batch = 2048
        self.max_timesteps_per_episode = 1000

    def learn(self, total_steps):
        current_step = 0
        while current_step < total_steps:
            pass
        
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
            pass