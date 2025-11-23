import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        self.hidden_size = 128

        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc4 = nn.Linear(self.hidden_size, output_size)
    
    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def forward_actor(self, x):
        x = self.forward(x)
        x = torch.tanh(x)
        return x
    
    def forward_critic(self, x):
        x = self.forward(x)
        return x
